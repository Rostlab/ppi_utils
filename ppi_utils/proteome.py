import re
import shutil
from pathlib import Path
from typing import Union, Iterable

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from ppi_utils.general import run_uniqueprot, run_uniqueprot2D, get_seq_hash, glob_type


def filter_proteomes(fasta_dir: Path,
                     min_len: int = 50, max_len: int = 1500,
                     blacklist_fasta: Union[Path, Iterable[Path]] = None
                     ) -> set[int]:
    spp = set()
    if blacklist_fasta is None:
        blacklist = set()
    elif hasattr(blacklist_fasta, '__iter__'):
        blacklist = set().union(*({r.id for r in SeqIO.parse(fasta, 'fasta')}
                                  for fasta in blacklist_fasta))
    else:
        blacklist = {r.id for r in SeqIO.parse(blacklist_fasta, 'fasta')}
    for fasta in tqdm(glob_type(fasta_dir, '.fasta')):
        if not fasta.stem.isnumeric():
            continue
        spp.add(int(fasta.stem))
        bak = fasta.with_suffix('.fasta.bak')
        if not bak.is_file():
            # if there is no backup, make one
            shutil.move(fasta, bak)
        # if there was or is now, fall back to it anyway
        records = SeqIO.to_dict(SeqIO.parse(bak, 'fasta'))
        with fasta.open('w') as handle:
            for _id in sorted(records.keys()):
                record = records[_id]
                if len(record) < min_len or len(record) > max_len or _id in blacklist:
                    continue
                SeqIO.write(record, handle, 'fasta')
    return spp


def rr_pattern_proteomes(species: set[int],
                         proteome_dir: Path,
                         pattern: str = '{sp}.fasta',
                         prefix: str = '',
                         ref_fasta: Path = None,
                         hval_config: Path = None,
                         pretend: bool = False) -> Union[str, None]:
    cmd_lines = list()
    with tqdm(sorted(species)) as pbar:
        for sp in pbar:
            pbar.set_postfix(batch=sp)
            sp_fasta = proteome_dir / pattern.format(sp=sp)
            assert sp_fasta.is_file(), f'Missing {sp_fasta}'
            if ref_fasta is not None:
                cmd_line = run_uniqueprot2D(
                    sp_fasta, ref_fasta,
                    sp_fasta.with_stem(f'{sp}_nr_{prefix}'),
                    hval_config, pretend)
            else:
                cmd_line = run_uniqueprot(
                    sp_fasta,
                    sp_fasta.with_stem(f'{sp}_rr_{prefix}'),
                    hval_config, pretend)
            cmd_lines.append(cmd_line)
    if pretend:
        return ' && '.join(cmd_lines)


def merge_pattern_proteomes(species: set[int],
                            merged_fasta: Path,
                            proteome_dir: Path,
                            pattern: str = '{sp}.fasta') -> None:
    with merged_fasta.open('w') as merged:
        for sp in sorted(species):
            sp_fasta = proteome_dir / pattern.format(sp=sp)
            assert sp_fasta.is_file(), f'Missing {sp_fasta}'
            with sp_fasta.open('r') as single:
                merged.write(single.read())


def read_hash_proteomes(species: set[int], proteome_dir: Path,
                        pattern: str) -> dict[int, dict[str, dict[str, str]]]:
    records = dict()
    with tqdm(sorted(species)) as pbar:
        for sp in pbar:
            pbar.set_postfix(batch=sp)
            sp_fasta = proteome_dir / pattern.format(sp=int(sp))
            assert sp_fasta.is_file(), f'Missing {sp_fasta}'

            sp_dict = dict()
            for record in SeqIO.parse(sp_fasta, 'fasta'):
                crc = get_seq_hash(record.seq)
                sp_dict[crc] = dict(seq=str(record.seq), id=record.id,
                                    description=record.description)
            records[int(sp)] = sp_dict
    return records


def parse_val_proteome(fasta: Path) -> dict[int, dict[str, SeqRecord]]:
    # '(?P<db>(?:sp|tr))\|(?P<accession>.+?)\|(?P<name>\S+?) (?P<full_name>.+?) OS=(?P<organism>.+?) OX=(?P<taxon_id>.+?) (GN=(?P<gene>.+?) )?PE=(?P<evidence_level>.+?) SV=(?P<version>.+?)$'
    records = dict()
    taxon_regex = re.compile('.+\sOX=(?P<taxon_id>.+?)\s.+')
    assert fasta.is_file()
    for record in tqdm(SeqIO.parse(fasta, 'fasta')):
        m = taxon_regex.match(record.description)
        assert m, f'No match: {record.description}'
        sp = int(m.groupdict()['taxon_id'])
        sp_dict = records.get(sp, dict())
        crc = get_seq_hash(record.seq)
        sp_dict[crc] = dict(seq=str(record.seq), id=record.id,
                            description=record.description)
        records[sp] = sp_dict
    return records


def get_proteome_seqs(proteome: dict[int, dict[str, dict]],
                      fasta: dict[str, SeqRecord],
                      negatives: pd.DataFrame) -> dict[str, SeqRecord]:
    negatives = negatives.copy().melt(
        id_vars=['species', 'label'], value_name='crc_hash')[['species', 'crc_hash']]
    extra_crcs = negatives.loc[~negatives.crc_hash.isin(
        fasta.keys()), ['crc_hash', 'species']]
    swissprot_seqs = dict()
    for sp, crcs in extra_crcs.groupby('species'):
        for crc in crcs.crc_hash:
            entry = proteome[sp][crc]
            swissprot_seqs[crc] = SeqRecord(seq=Seq(entry['seq']),
                                            id=crc, name=crc, description=crc)
    return swissprot_seqs
