#!/usr/bin/env python3
import io
import json
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ppi_utils.general import glob_type
from ppi_utils.reduce import dedup_pairs


def unzip_apid(zip_path: Union[str, Path] = None,
               work_dir: Union[str, Path] = '.',
               keep_human: bool = False,
               keep_interspecies: bool = False,
               ) -> Path:
    if zip_path is None:
        zip_path = Path('apid.zip')
    else:
        zip_path = Path(zip_path)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    zd = work_dir / 'apid_q1'

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(zd)
    if keep_interspecies:
        [d.unlink() for d in zd.glob('*_noISI_Q1.txt')]
    else:
        [d.unlink() for d in set(zd.glob('*.txt')) - set(zd.glob('*_noISI_Q1.txt'))]
    if not keep_human:
        f = next(zd.glob('9606_*Q1.txt'))
        f.rename(f.with_suffix('.txt.bak'))
    return zd


def _extract_uniprots_from_apid_tsv(
        q1_path: Path) -> set:
    q1_df = pd.read_csv(q1_path, sep='\t', header=0)
    q1_ids = set(q1_df[['UniprotID_A', 'UniprotID_B']]
                 .values.flatten())
    return q1_ids


def extract_apid_uniprot_ids(
        apid_dir: Union[str, Path] = Path('apid'),
        q_level: int = 1) -> set:
    apid_dir = Path(apid_dir)
    return set.union(*(
        _extract_uniprots_from_apid_tsv(f)
        for f in glob_type(apid_dir, f'_Q{q_level}.txt')))


def _extract_pairs_from_apid_tsv(
        q1_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read in an APID _Qx.txt as a TSV, extract the two UniProt ID columns
    and return them as a pd of sorted pairs.
    :param q1_path:
    :return:
    """
    q1_path = Path(q1_path)
    q1_df = pd.read_csv(q1_path, sep='\t', header=0)[['UniprotID_A', 'UniprotID_B']]
    q1_df['species'] = q1_path.stem.split('_')[0]
    q1_df.species = q1_df.species.astype('int64')
    return dedup_pairs(q1_df)


def extract_apid_ppis(
        apid_dir: Union[str, Path] = Path('apid'),
        q_level: int = 1) -> pd.DataFrame:
    """
    Extract pairs from all APID _Qx.txt tables, concat, de-duplicate and return.
    :param apid_dir:
    :param q_level
    :return:
    """
    apid_dir = Path(apid_dir)
    return pd.concat([_extract_pairs_from_apid_tsv(f)
                      for f in glob_type(apid_dir, f'_Q{q_level}.txt')]) \
        .drop_duplicates()


def extract_huri_ppis(psi_path: Union[str, Path]) -> tuple[pd.DataFrame, dict]:
    huri = pd.read_csv(psi_path, sep='\t', header=None).iloc[:, [0, 1]]

    all_ids = {'ensembl': set(), 'uniprotkb': set()}
    l = lambda s: s.split('.')[0]
    [all_ids[db].add(l(_id)) for db, _id in [entry.split(':') for entry in np.unique(huri)]]

    extract = np.vectorize(lambda s: s.split(':')[1].split('.')[0])
    huri = pd.DataFrame(extract(huri)).drop_duplicates()
    huri['species'] = 9606  # the taxid for H. sapiens
    return huri, all_ids


def ppis_from_string(raw: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(raw), sep='\t', header=None)


def ppis_to_hashes(ppis: pd.DataFrame, json_path: Path,
                   horizontal_sort: bool = True,
                   dedup: bool = False) -> pd.DataFrame:
    with json_path.open('r') as json_file:
        lookup = json.load(json_file)
    ppis = ppis.copy()
    _to_hash = np.vectorize(lookup.get)
    ppis.iloc[:, [0, 1]] = _to_hash(ppis.iloc[:, [0, 1]])
    ppis.columns = ['hash_A', 'hash_B'] + list(ppis.columns)[2:]
    if 'species' in ppis.columns:
        ppis.species = ppis.species.astype('int64')
    if horizontal_sort:
        ppis = dedup_pairs(ppis)
    ppis = ppis.sort_values(
        by=['hash_A', 'hash_B']
           + (['species'] if 'species' in ppis.columns else []))
    if dedup:
        ppis = ppis.drop_duplicates().reset_index(drop=True)
    return ppis
