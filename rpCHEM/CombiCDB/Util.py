#!/usr/bin/env python
"""Miscellaneous utility functions used across the application
"""
from rpCHEM.CombiCDB import Const, Env
import sys, os
import logging
from optparse import OptionParser

from rpCHEM.Common.Util import createStandardSmiString, molBySmiles;

log = logging.getLogger(Const.APPLICATION_NAME)
log.setLevel(Const.LOGGER_LEVEL)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(Const.LOGGER_FORMAT)

handler.setFormatter(formatter)
log.addHandler(handler)

def readIDFile( idFile ):
    """Given a file of database IDs, return a list of the IDs.
    Expect one ID per line of the file.
    The file can contain other contents, the ID value just has to be the last
    whitespace-separated item on each line.  The rest of the contents will be ignored.
    """
    idList = []
    for line in idFile:
        chunks = line.split()
        if len(chunks) > 0: # Skip blank lines
            idList.append(chunks[-1])
    return idList

def smilesListToCanonicalSet(smiListStr):
    """Utility function to split a smiles list str, canonicalize, and return a set of the smiles"""
    smiList = smiListStr.split(Const.MOL_LIST_DELIM);
    molList = [molBySmiles(smi) for smi in smiList];
    return set([createStandardSmiString(mol) for mol in molList]);


def main(argv):
    """Main method, callable from command line"""
    pass

if __name__ == "__main__":
    main(sys.argv)
    
