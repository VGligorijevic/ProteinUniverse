#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
from pathlib import Path

import torch

from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder

def make_distance_map(pdbfile):
    """
    Generate (diagonalized) C𝛼 distance matrix from a pdbfile 
    """
    with open(pdbfile, 'r') as f:
        structure_container = build_structure_container_for_pdb(f.read()) 
        distance_map  = DistanceMapBuilder().generate_map_for_pdb(structure_container)
    return distance_map.chains

def exists(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    return path


def input_yielder(path):
    """Crafts a generator that iterates through every file
    under the path if its a dir or yields the file itself otherwise
    """
    path = exists(path)
    if path.is_file():
        def yielder():
            yield path
    elif path.is_dir():
        def yielder():
            yield from path.absolute().iterdir()
    return yielder()

def arguments():
    parser = argparse.ArgumentParser(description="Save PDB file(s) as distance matrices")
    parser.add_argument("-i",
                        type=input_yielder,
                        dest='inputs', nargs='+',
                        required=True,
                        help="Input director(y|ies) and pdb file(s).")

    parser.add_argument("-o",
                        type=Path,
                        dest='outputs',
                        default='distance_maps',
                        help="Output directory to dump tensor files")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    args.outputs.mkdir(exist_ok=True, parents=True)
    (args.outputs / "tensors").mkdir(exist_ok=True, parents=True)
    (args.outputs / "fastas").mkdir(exist_ok=True, parents=True)

    for pdbfile in itertools.chain.from_iterable(args.inputs):
        dmap  = make_distance_map(pdbfile)

        stem  = pdbfile.stem
        chain = list(dmap.keys())[0] 
        fsa  = args.outputs / 'fastas' / f"{stem}.fsa"
        tnsr = args.outputs / 'tensors' / f"{stem}.pt"
        chainmap = dmap[chain]['contact-map']
        seq      = str(dmap[chain]['seq'])

        with open(fsa, 'w') as fsh:
            print(f">{stem}", file=fsh)
            print(seq, file=fsh)

        torch.save(torch.from_numpy(chainmap), tnsr) 
