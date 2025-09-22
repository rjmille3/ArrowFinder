#!/usr/bin/env python3
# encoding: utf-8
"""
Evaluate source/sink + ranker with *no parallel model inference* (CLI version).

Workers:
  - Build orb-pair candidates from top-K source/sink atoms
  - Filter to ops that recover the *products* (to cut payload size)
  - Return serialized ops and whether any arrows-only matches existed

Main process:
  - Loads all models (source, sink, ranker)
  - Does source/sink inference (per reaction) BEFORE multiprocessing
  - Dispatches lightweight tasks to workers for op generation
  - Serially ranks product-matching ops with the ranker (no GPU contention)
  - Writes one output row per reaction

Output CSV columns:
  index, status, input_rxn, target_products, matched_prediction, products_match, arrows_match, note

status ∈ {"recovered_both", "recovered_products", "recovered_arrows", "not_recovered", "error"}
"""

import os, sys, csv, h5py, time, argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from tensorflow.keras.models import load_model

from rdkit import Chem
from openeye.oechem import OEMol, OESmilesToMol, OECreateSmiString

from reaction_prediction.atom.modules.path_extractor import PathExtractor as PE
from reaction_prediction.atom.modules.feature_extraction import FeatureExtraction as FE
from reaction_prediction.ranker.modules.feature_extraction import ReactionFeatureExtraction as RFE
from reaction_prediction.ranker.modules.simple_orbpair_object import SimpleOrbPairObject as SOO
from reaction_prediction.atom.modules.simple_atom_object import SimpleAtomObject as SAO


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate source/sink + ranker on reaction inputs (no parallel model inference).")
    # Required paths
    p.add_argument("--input",  required=True, help="Path to input reactions CSV/TXT (one reaction per line/row[0])")
    p.add_argument("--output", required=True, help="Output CSV path for results")
    p.add_argument("--allid",  required=True, help="Path to allids JSON for feature extraction")

    # Required models
    p.add_argument("--source_model", required=True, help="Path to source atom model (.h5)")
    p.add_argument("--sink_model",   required=True, help="Path to sink atom model (.h5)")
    p.add_argument("--ranker_model", required=True, help="Path to Siamese/Ranker model (.h5)")

    p.add_argument("--ranker_inner_index", type=int, default=2,
                   help="Fallback layer index if name not found (default: 2)")

    # Optional knobs
    p.add_argument("--max_orbs", type=int, default=5000, help="Max orbital pairs to consider per (src,snk) (default: 5000)")
    p.add_argument("--top_k",    type=int, default=10,   help="Top-K atoms to keep for source/sink (default: 10)")
    p.add_argument("--chunk_size", type=int, default=512, help="How many reactions to process per chunk (default: 512)")
    p.add_argument("--n_procs", type=int, default=max(1, cpu_count()-1),
                   help="Worker processes for op generation (default: CPU-1)")
    p.add_argument("--pool_chunksize", type=int, default=8,
                   help="imap_unordered chunksize for worker tasks (default: 8)")

    return p.parse_args()


# ---------------------------
# Utilities
# ---------------------------

def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

def patch_keras_h5_training_config(h5_path: str):
    """Fix older Keras 'learning_rate' key -> 'lr' so load_model doesn't choke."""
    with h5py.File(h5_path, 'r+') as f:
        if 'training_config' in f.attrs:
            data = f.attrs['training_config']
            if isinstance(data, (bytes, bytearray)) and b'learning_rate' in data:
                f.attrs['training_config'] = data.decode().replace("learning_rate", "lr").encode()

def canonicalize_smiles_nomaps_oechem(smiles: str) -> str:
    """Return canonical SMILES with all atom-maps removed using OpenEye."""
    mol = OEMol()
    if not OESmilesToMol(mol, smiles):
        return ""
    for atom in mol.GetAtoms():
        atom.SetMapIdx(0)
    return OECreateSmiString(mol)

def extract_atom_fv_to_numpy_array(atom_list, allid_file):
    """Given list of per-atom connectedSmiles strings, return stacked feature vectors."""
    pe = PE(3)
    fe = FE(allid_file, pe)
    fv_list = []
    for atom in atom_list:
        fv, _ = fe.atom_feat_vec(atom)
        fv_list.append(fv)
    return np.asarray(fv_list)

def parse_reaction_line(input_rxn: str):
    """
    Expected formats:
      "<reactants> >> <products> <arrows...>"
      or "<reactants> >> <products>"
    Returns (reactants, canon_products, arrows_expected)
    """
    rxn_full = input_rxn
    reactants = rxn_full.split(">>")[0]
    rxn_only  = rxn_full.split(" ")[0]
    parts     = rxn_full.split(" ")
    arrows_expected = parts[1].strip().rstrip(",") if len(parts) > 1 else ""
    products = rxn_only.split(">>")[1].strip()
    canon_products = canonicalize_smiles_nomaps_oechem(products)
    return reactants, canon_products, arrows_expected

def pick_topk_indices(scores: np.ndarray, k: int) -> list:
    """Return indices of the top-k values (descending)."""
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k-1)[:k]
    idx = idx[np.argsort(scores[idx])[::-1]]  # sort those top-k indices by score desc
    return idx.tolist()


def get_atoms_list(reactants: str):
    return list(SAO.atomObjFromReactantSmi(reactants))

# ---------------------------
# Ranker helpers (main process only)
# ---------------------------

def extract_single_op_fv_to_list_serialized(op_ser, allid_file):
    """
    Feature row for a serialized op:
      op_ser = {'rxn','src','snk','arrows','prod'}
    """
    rfe = RFE(morgan_radi=2, morgan_bits=2048, atom_length=3, allid_file=allid_file)
    return rfe.extract_rxn_rep(op_ser['rxn'], op_ser['src'], op_ser['snk'])


# ---------------------------
# Worker: *no* model inference here
# ---------------------------

def _serialize_op(op):
    return {
        'rxn': op.reactionSmiles,
        'src': op.srcAtom.connectedSmiles,
        'snk': op.sinkAtom.connectedSmiles,
        'arrows': op.arrowCodes,
        'prod': op.productSmiles,
    }

def worker_build_ops(task):
    """
    task = {
      'global_idx': int,
      'input_rxn': str,
      'reactants': str,
      'canon_products': str,
      'arrows_expected': str,
      'src_idx': List[int],
      'snk_idx': List[int],
      'max_orbs': int,
    }

    Returns:
      (global_idx, {'product_ops': [serialized_op,...], 'arrows_only': bool})
      or
      (global_idx, {'error': '...'})
    """
    i = task['global_idx']
    try:
        reactants = task['reactants']
        canon_products = task['canon_products']
        arrows_expected = task['arrows_expected']
        src_idx = task['src_idx']
        snk_idx = task['snk_idx']
        max_orbs = task['max_orbs']

        atoms = get_atoms_list(reactants)
        source_list = [atoms[j] for j in src_idx]
        sink_list   = [atoms[j] for j in snk_idx]

        ops = SOO.orbPairObjectsFromAtoms_bounded(
            source_list, sink_list, max_orbs=max_orbs, radical=False
        )

        product_ops_ser = []
        arrows_only = False

        for op in ops:
            canon_pred = canonicalize_smiles_nomaps_oechem(op.productSmiles)
            if bool(canon_pred) and (canon_pred == canon_products):
                product_ops_ser.append(_serialize_op(op))
            elif op.arrowCodes.strip() == arrows_expected.strip():
                arrows_only = True

        return i, {'product_ops': product_ops_ser, 'arrows_only': arrows_only}

    except Exception as e:
        return i, {'error': str(e).replace("\n", " ")}


# ---------------------------
# Core eval (main process does parsing + all model inference)
# ---------------------------

def run_eval(
    input_file: str,
    allid_file: str,
    output_csv: str,
    source_model_path: str,
    sink_model_path: str,
    ranker_model_path: str,
    ranker_inner_index: int,
    max_orbs: int,
    topk: int,
    chunk_size: int,
    n_procs: int,
    pool_chunksize: int,
):
    t_start = time.time()
    print(f"[init] Loading models...", flush=True)

    # Patch & load models (main process only)
    for pth in (source_model_path, sink_model_path, ranker_model_path):
        patch_keras_h5_training_config(pth)

    source_model = load_model(source_model_path)
    sink_model   = load_model(sink_model_path)

    # Load the Siamese and get the shared ranker submodel in MAIN process
    siamese_model = load_model(ranker_model_path)
    ranker_model = siamese_model.layers[ranker_inner_index]

    print(f"[init] Models loaded. Using N_PROCS={n_procs}, CHUNK_SIZE={chunk_size}", flush=True)

    # Ensure output dir
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # CSV writer
    with open(output_csv, "w", newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "index", "status", "input_rxn", "target_products",
            "matched_prediction", "products_match", "arrows_match", "note"
        ])

        # Stream input, process in CHUNK_SIZE batches
        with open(input_file, newline='') as f:
            reader = csv.reader(f)

            buffer = []  # list of (global_idx, input_rxn)
            total_processed = 0
            chunk_id = 0

            print(f"[run] Beginning streaming from: {input_file}", flush=True)

            for i, row in enumerate(reader):
                input_rxn = row[0] if row else ""
                buffer.append((i, input_rxn))

                if len(buffer) >= chunk_size:
                    chunk_id += 1
                    t0 = time.time()
                    n = process_chunk(
                        buffer, source_model, sink_model, ranker_model, writer,
                        allid_file, max_orbs, topk, n_procs, pool_chunksize
                    )
                    dt = time.time() - t0
                    total_processed += n
                    elapsed = time.time() - t_start
                    rps = (total_processed / elapsed) if elapsed > 0 else 0.0
                    print(f"[chunk {chunk_id}] processed {n} rxns in {dt:.2f}s | "
                          f"total {total_processed} | elapsed {_fmt_hms(elapsed)} | "
                          f"{rps:.2f} rxn/s",
                          flush=True)
                    buffer = []

            # tail
            if buffer:
                chunk_id += 1
                t0 = time.time()
                n = process_chunk(
                    buffer, source_model, sink_model, ranker_model, writer,
                    allid_file, max_orbs, topk, n_procs, pool_chunksize
                )
                dt = time.time() - t0
                total_processed += n
                elapsed = time.time() - t_start
                rps = (total_processed / elapsed) if elapsed > 0 else 0.0
                print(f"[chunk {chunk_id}] processed {n} rxns in {dt:.2f}s | "
                      f"total {total_processed} | elapsed {_fmt_hms(elapsed)} | "
                      f"{rps:.2f} rxn/s",
                      flush=True)

    total_elapsed = time.time() - t_start
    print(f"[done] Output: {output_csv}", flush=True)
    print(f"[done] Total processed: {total_processed} | Total elapsed: {_fmt_hms(total_elapsed)} "
          f"| Avg throughput: {(total_processed/total_elapsed if total_elapsed>0 else 0.0):.2f} rxn/s",
          flush=True)


def process_chunk(
    buffer, source_model, sink_model, ranker_model, writer,
    allid_file, max_orbs, topk, n_procs, pool_chunksize
):
    """
    buffer: list of (global_idx, input_rxn)
    Steps:
      1) parse + canonicalize targets  (main)
      2) atom features + source/sink inference (main)
      3) build tasks (main)
      4) worker_build_ops to generate serialized product-matching ops (pool)
      5) serial ranker inference & write rows (main)
    """
    tasks = []

    # 1–3: prepare tasks with top-K atom INDICES (no model inference in workers)
    for (global_idx, input_rxn) in buffer:
        try:
            reactants, canon_products, arrows_expected = parse_reaction_line(input_rxn)

            atoms = get_atoms_list(reactants)
            if not atoms:
                tasks.append({
                    'global_idx': global_idx,
                    'input_rxn': input_rxn,
                    'reactants': reactants,
                    'canon_products': canon_products,
                    'arrows_expected': arrows_expected,
                    'src_idx': [],
                    'snk_idx': [],
                    'max_orbs': max_orbs,
                    '__force_error__': "No atoms parsed from reactants."
                })
                continue

            atoms_oesmiles = [atom.connectedSmiles for atom in atoms]
            atoms_feature_array = extract_atom_fv_to_numpy_array(atoms_oesmiles, allid_file)

            # Predict source/sink (main process)
            src_scores = source_model.predict(atoms_feature_array, verbose=0).ravel()
            snk_scores = sink_model.predict(atoms_feature_array,   verbose=0).ravel()

            src_idx = pick_topk_indices(src_scores, topk)
            snk_idx = pick_topk_indices(snk_scores, topk)

            tasks.append({
                'global_idx': global_idx,
                'input_rxn': input_rxn,
                'reactants': reactants,
                'canon_products': canon_products,
                'arrows_expected': arrows_expected,
                'src_idx': src_idx,
                'snk_idx': snk_idx,
                'max_orbs': max_orbs,
            })

        except Exception as e:
            tasks.append({
                'global_idx': global_idx,
                'input_rxn': input_rxn,
                'reactants': "",
                'canon_products': "",
                'arrows_expected': "",
                'src_idx': [],
                'snk_idx': [],
                'max_orbs': max_orbs,
                '__force_error__': str(e).replace("\n", " "),
            })

    # Split out direct error rows
    direct_rows = []
    real_tasks  = []
    for t in tasks:
        if '__force_error__' in t:
            i = t['global_idx']
            input_rxn = t['input_rxn']
            note = t['__force_error__']
            row = [i, "error", input_rxn, "", "", 0, 0, note]
            direct_rows.append((i, row))
        else:
            real_tasks.append(t)

    # 4) Generate ops in parallel (NO model inference here)
    results_map = {}
    if real_tasks:
        print(f"  [pool] dispatching {len(real_tasks)} tasks to {n_procs} workers...", flush=True)
        t_pool0 = time.time()
        with Pool(processes=n_procs) as pool:
            for gi, payload in pool.imap_unordered(worker_build_ops, real_tasks, chunksize=pool_chunksize):
                results_map[gi] = payload
        t_pool = time.time() - t_pool0
        print(f"  [pool] completed {len(real_tasks)} tasks in {t_pool:.2f}s", flush=True)

    # 5) Serial ranker inference + write rows (in original order)
    all_rows = []

    # Map tasks by idx for ordered pass
    task_map = {t['global_idx']: t for t in real_tasks}
    # Include direct errors
    for i, row in direct_rows:
        all_rows.append((i, row))

    for gi in sorted(task_map.keys()):
        t = task_map[gi]
        payload = results_map.get(gi, {'error': 'Missing worker result'})
        input_rxn = t['input_rxn']
        canon_products = t['canon_products']
        arrows_expected = t['arrows_expected']

        # Defaults
        status = "not_recovered"
        matched_prediction = ""
        products_match_flag = 0
        arrows_match_flag   = 0
        note = ""

        if 'error' in payload:
            status = "error"
            note = payload['error']
        else:
            product_ops = payload.get('product_ops', [])
            arrows_only = bool(payload.get('arrows_only', False))

            if not product_ops:
                if arrows_only:
                    status = "recovered_arrows"
                    arrows_match_flag = 1
                else:
                    status = "not_recovered"
                    if not canon_products:
                        note = "Target products could not be canonicalized."
            else:
                # Build feature matrix serially, run ranker serially
                ops_feature_rows = []
                valid_ops = []
                for op_ser in product_ops:
                    try:
                        fv = extract_single_op_fv_to_list_serialized(op_ser, allid_file)
                        ops_feature_rows.append(fv)
                        valid_ops.append(op_ser)
                    except Exception:
                        # skip op
                        continue

                if not valid_ops:
                    status = "not_recovered"
                    note = "No valid feature rows for product-matching ops."
                else:
                    ops_scores = np.asarray(
                        ranker_model.predict(np.array(ops_feature_rows), verbose=0)
                    ).reshape(-1)

                    top_idx = int(np.argmax(ops_scores))
                    top_op = valid_ops[top_idx]

                    matched_prediction = f"{top_op['rxn']} {top_op['arrows']}".strip()
                    products_match_flag = 1
                    arrows_match_flag   = 1 if top_op['arrows'].strip() == arrows_expected.strip() else 0
                    status = "recovered_both" if arrows_match_flag == 1 else "recovered_products"

        all_rows.append((
            gi,
            [
                gi,
                status,
                input_rxn,
                canon_products,
                matched_prediction,
                products_match_flag,
                arrows_match_flag,
                note
            ]
        ))

    # Write in order
    all_rows.sort(key=lambda x: x[0])
    for _, row in all_rows:
        writer.writerow(row)

    # Return count
    return len(buffer)


# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_eval(
        input_file=args.input,
        allid_file=args.allid,
        output_csv=args.output,
        source_model_path=args.source_model,
        sink_model_path=args.sink_model,
        ranker_model_path=args.ranker_model,
        ranker_inner_index=args.ranker_inner_index,
        max_orbs=args.max_orbs,
        topk=args.top_k,
        chunk_size=args.chunk_size,
        n_procs=args.n_procs,
        pool_chunksize=args.pool_chunksize,
    ))
