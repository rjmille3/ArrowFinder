# ArrowFinder and Two-Stage Reaction Prediction

This repository provides scripts and models for atom and mechanism ranking used in the ArrowFinder and Two-Stage models.

Additional data splits and model checkpoints used in the PMechRP paper can be found at https://deeprxn.ics.uci.edu/pmechdb/download under "PMechRP Datasets" and "PMechRP Model Checkpoints".

---

## Setup Python Environment

```bash
# Create conda environment
conda create -n two_stage python=3.7.0

# Install OpenEye Toolkits (requires license)
conda install openeye::openeye-toolkits

# Install requirements
pip install -r requirements.txt
```

---

## Update PYTHONPATH

Add the working directory and `CombiCDB` folder to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/rpCHEM/CombiCDB"
```

---

## OpenEye License

This repository depends on **OpenEye Toolkits**, which require a valid license.

```bash
# Export OE_LICENSE
export OE_LICENSE="path/to/oe_license.txt"
```

- The OpenEye Academic License Agreement is available at: `openeye_license_agreement.txt`  
- Although the source code is released under the MIT license, any use of OpenEye libraries must comply with their licensing terms.

---

## Atom Model Workflow

1. **Create graph topological features**  
   ```bash
   shell_scripts/datagen/create_path_features.sh
   ```
   Reactions must be formatted with quotation marks, e.g.:

   ```text
   "[Li:10][O+:11](C)C1=CC=CC=C1>>[Li+:10].C[O:11]C1=CC=CC=C1 10,11=11"
   ```

2. **Generate training files for sink and source models**  
   ```bash
   shell_scripts/datagen/create_sink.sh
   shell_scripts/datagen/create_source.sh
   ```

3. **Train atom models**  
   ```bash
   shell_scripts/training/train_sink.sh
   shell_scripts/training/train_source.sh
   ```

---

## Ranker Model Workflow

1. **Create training files**  
   ```bash
   shell_scripts/datagen/create_ranker.sh
   ```
   Input format:  
   ```text
   rxn, arrow, source, sink
   ```
   If needed, reformat with:  
   ```bash
   shell_scripts/datagen/reformat_rxns.sh
   ```

2. **Train the ranker model**  
   ```bash
   shell_scripts/training/train_ranker.sh
   ```

---

## Evaluation

1. **Generate two-stage predictions**  
   ```bash
   shell_scripts/eval/run_two_stage_prediction.sh
   ```
   This produces:
   - `bad.csv` → lines that failed processing  
   - `preds_ascending.csv` → predictions sorted by ascending score  
   - `preds_descending.csv` → predictions sorted by descending score  

   ⚠️ The model learns relative score differences between plausible and implausible mechanisms. The score orientation is not guaranteed, so check both ascending and descending predictions.

2. **Compute accuracy**  
   ```bash
   shell_scripts/eval/compute_accuracy.sh
   ```

---

## Generate ArrowFinder predictions

Run ArrowFinder predictions with:

```bash
shell_scripts/eval/run_arrowfinder_prediction.sh
```

The script reports five categories of recovery:

- **recovered_both** – cases where both the products and arrows are recovered.
- **recovered_products** – cases where only the products are recovered, but the arrows differ.
- **recovered_arrows** – cases where only the arrow codes are recovered, but the products differ.
- **not_recovered** – cases where no mechanism was found.
- **error** – cases where the reactants could not be processed.
  

From our analysis on the mixed fold0 predictions, we found that in all cases where products were recovered, the predicted arrows (even if they did not exactly match the ground-truth arrows) were still correct.  

Therefore, we define the **accuracy** as:

`accuracy = (recovered_both + recovered_products) / total_reactions`
