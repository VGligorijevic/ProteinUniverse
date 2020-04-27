#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to set up computing intra-topology TMAlign scores amongst 
cath domains. 
"""
import itertools
import operator as op
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ProteinUniverse      = Path("/mnt/home/dberenberg/projects/ProteinUniverse")
TMAlign              = ProteinUniverse / "dependencies/bin/TMAlign"

CATH                 = ProteinUniverse / "Data/cath"
CATH_STRUCTURES      = CATH / "cath-dataset-nr-S40"
CATH_CLASSIFICATION  = CATH / 'materials/metadata/domain-classifications.tsv'

def retrieve_structure_file_by_id(structure_id):
    candidate = CATH_STRUCTURES / f"{structure_id}.pdb"
    if not candidate.exists():
        return None
    return candidate.absolute()


def star_eq(args):
    return op.eq(*args)

def progress(curr:int,
             tot:int,
             total_len:int = 80,
             barchr:str='#',
             currchr:str='@', 
             emptychr:str='=', 
             percent:bool=True) -> str:
    """
    Make a progress bar.
    
    args:
        :curr (int) - current index
        :tot  (int) - total length of run
        :total_len (int) - length of progress bar
        :barchr (str) - char for finished progress
        :currchr (str) - char for current progress
        :emptychr (str) - char for unfinished progress
        :percent (bool) - include percent done at end of pbar
    returns:
        :(str) - constructed progress bar
    """
    if isinstance(currchr, (list, tuple)): # rotating progress bar
        currchr = currchr[curr % len(currchr)]
        
    prog  = (curr + 1)/tot
    nbars = int(total_len * prog) - 1
    rest  = total_len - nbars - 1
    bar = f"[{nbars * barchr}{currchr if rest else ''}{emptychr * rest}]"
    return bar + f"({prog * 100:0.2f}%)" if percent else bar

CLEAR = f"\r{80 * ' '}\r"

if __name__ == '__main__':
    cath_annotation_frame = pd.read_table(CATH_CLASSIFICATION)
    ct = 0
    K = 80 # sample this amount from each TOPOL


    available_domains = list(map(lambda dom: dom.stem, CATH_STRUCTURES.iterdir()))
    cath_annotation_frame = cath_annotation_frame[cath_annotation_frame.DOMAIN.isin(available_domains)]
    N = cath_annotation_frame.TOPOL.unique().size
    print(f"[*] Found {len(cath_annotation_frame)} avail. domains ({N} topologies)")
    print(f"[*] Choosing at most {K} domains per topology.")
    cts = np.zeros(N)
    
    output_root = CATH / "intratopology-tmscores"

    dirct = dirnum = 0 # number of files per dir, dir ID resp.
    output_dir  = output_root / f"{dirnum:03d}"
    max_per_dir = 100_000
    output_dir.mkdir(exist_ok=True, parents=True)
    
    start = datetime.now()
    last_el = None
    with open("cath_intratopol_tmscore.tsk", 'w') as tmsc:
        for i, (topology, frame) in enumerate(cath_annotation_frame.groupby("TOPOL")):
            frame = frame.sample(K).copy() if len(frame) > K else frame
            elapsed = datetime.now() - start 
            substart = datetime.now()
            #oprint(f"{progress(i, N, 60)} {topology} {len(frame)}", flush=True)
            print(topology, len(frame), flush=True)
            #({elapsed} since start, last took {last_el})", end='', flush=True) 
            for j, domains in enumerate(itertools.filterfalse(star_eq, itertools.product(frame.DOMAIN.values, frame.DOMAIN.values))):
                d1, d2 = domains
                pdb1, pdb2 = map(retrieve_structure_file_by_id, domains)
                if dirct >= max_per_dir:
                    dirnum += 1
                    dirct = 0
                    output_dir = output_root / f"{dirnum:03d}" 
                    output_dir.mkdir(exist_ok=True, parents=True)   

                outfile = output_dir / f"{d1}_{d2}.tmscore"
                dirct += 1

                tmsc.write(f"{TMAlign} {pdb1} {pdb2} > {outfile}\n")
                print(f"\t{CLEAR}{j}/{len(frame)*(len(frame) - 1)}",end='', flush=True)

                cts[i] += 1
            last_el = datetime.now() - substart
            print90
            
    
    metrics = set(['sum', 'median', 'mean', 'std'])
    values = [getattr(np, metric)(cts) for metric in metrics]
    mvd = dict(zip(metrics, values))
    tot = mvd['sum']

    print(f"{CLEAR}[*] Done! ({tot:,} total pairs, {datetime.now() - start} total elapsed.)")
    for metric in metrics - {'sum'}:
        print(f"{metric} pairs: {mvd[metric]:5.2f}")
    


