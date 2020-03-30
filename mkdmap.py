#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
from pathlib import Path

import torch

from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder

def make_distance_maps(pdbfile, sequence=None):
    """
    Generate (diagonalized) C𝛼 and C𝛽 distance matrix from a pdbfile 
    """
    pdb_handle = open(pdbfile, 'r')
    #with open(pdbfile, 'r') as f:
    #print(f)
    structure_container = build_structure_container_for_pdb(pdb_handle.read()).with_seqres(sequence) 
    
    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container) 
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()
    return ca.chains, cb.chains

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

def fasta_reader(handle, width=None):
    """
    Reads a FASTA file, yielding header, sequence pairs for each sequence recovered
    args:
        :handle (str, pathliob.Path, or file pointer) - fasta to read from
        :width (int or None) - formats the sequence to have max `width` character per line.
                               If <= 0, processed as None. If None, there is no max width.
    yields:
        :(header, sequence) tuples
    returns:
        :None
    """
    FASTA_STOP_CODON = "*"
    import io, textwrap, itertools

    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    width  = width if isinstance(width, int) and width > 0 else None
    try:
        for is_header, group in itertools.groupby(handle, lambda line: line.startswith(">")):
            if is_header:
                header = group.__next__().strip()
            else:
                seq    = ''.join(line.strip() for line in group).strip().rstrip(FASTA_STOP_CODON)
                if width is not None:
                    seq = textwrap.fill(seq, width)
                yield header, seq
    finally:
        if not handle.closed:
            handle.close()


def arguments():
    parser = argparse.ArgumentParser(description="Save PDB file(s) as distance matrices")
    parser.add_argument("-i",
                        type=input_yielder,
                        dest='inputs', nargs='+',
                        required=True,
                        help="Input director(y|ies) and pdb file(s).")
    
    parser.add_argument("-f",
                        type=Path,
                        dest="fasta",
                        help="Fasta to find seqres sequences (optional)")

    parser.add_argument("-o",
                        type=Path,
                        dest='outputs',
                        default='distance_maps',
                        help="Output directory to dump tensor files")
    return parser.parse_args()

def process_map(m):
    print(m)
    ch = list(m.keys())[0]

    seqres = m[ch].get('final-seq', None)
    seq    = m[ch]['seq']
    cmap   = m[ch]['contact-map']
    return seqres, seq, cmap

def write_fasta(header, sequence, filename):
    with open(filename, 'w') as f:
        print(f">{header}", file=f)
        print(sequence, file=f)

def process_cath_header(tup):
    h, s = tup
    return h.split("|")[2].split("/")[0], s

def write_tensor(filename, tensor):
    torch.save(torch.from_numpy(tensor), filename)

if __name__ == '__main__':
    args = arguments()
    if args.fasta is not None:
        seqres_map = dict(map(process_cath_header, fasta_reader(args.fasta))) 
    else:
        seqres_map = dict()

    args.outputs.mkdir(exist_ok=True, parents=True)
    tensors = args.outputs / 'tensors'
    fastas  = args.outputs / 'fastas'
    ca_out = tensors / 'ca'
    cb_out = tensors / 'cb'
    atom   = fastas / 'atom_lines'
    seqr   = fastas / 'seqres'
    
    ca_out.mkdir(parents=True, exist_ok=True)
    cb_out.mkdir(parents=True, exist_ok=True)
    atom.mkdir(parents=True, exist_ok=True)
    seqr.mkdir(parents=True, exist_ok=True)

    for pdbfile in itertools.chain.from_iterable(args.inputs):
        try:
            stem    = pdbfile.stem
            seqres  = seqres_map.get(stem, None)

            ca, cb  = make_distance_maps(pdbfile, sequence=seqres)
            (_, atom_line, ca_ca), (_, _, cb_cb) = map(process_map, (ca, cb))

            if seqres is not None:
                write_fasta(stem, seqres, seqr / f"{stem}.fsa")
            write_fasta(stem, atom_line, atom / f"{stem}.fsa")
            write_tensor(ca_out / f"{stem}.pt", ca_ca)
            write_tensor(cb_out / f"{stem}.pt", cb_cb)
        except Exception as e:
            with open(f"{stem}.err", 'w') as f:
                print(e, file=f)
