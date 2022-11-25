import json
import pickle
import warnings
from pathlib import Path
from typing import Union

import requests
from Bio import SeqIO
from tqdm import tqdm

from ppi_utils.general import get_seq_hash

warnings.warn(f'Module isn\'t up-to-date!: {Path(__file__)}', DeprecationWarning)


def parse_fasta(fasta_path: Union[str, Path], id_ref_path: Union[str, Path]) -> dict:
    print('Loading Sequences from: {}'.format(fasta_path))
    assert Path(fasta_path).is_file()

    seq_dict, identifiers = dict(), dict()

    with open(fasta_path, 'r') as fasta:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq.upper()
            seq_hash = get_seq_hash(seq)
            identifiers[record.id] = seq_hash
            if seq_hash not in seq_dict:
                seq_dict[seq_hash] = seq
            else:
                assert seq_dict[seq_hash] == seq, \
                    f'{fasta_path} contains a contradictory duplicate:\n' \
                    f'{record.format("fasta")}'

    save_identifiers_json(id_ref_path, identifiers)
    return seq_dict


def save_identifiers(path: Union[str, Path], identifiers: dict) -> None:
    print('Saving identifier references')
    with open(Path(path).with_suffix('.pkl'), 'wb') as f:
        pickle.dump(identifiers, f, pickle.HIGHEST_PROTOCOL)


def save_identifiers_json(path: Union[str, Path], identifiers: dict = None) -> None:
    p = Path(path)

    if identifiers is None:
        # translate an existing mapping from pickle to json
        assert p.is_file()
        identifiers = load_identifiers_json(p)

    # save identifiers as json
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix('.json'), 'w') as json_file:
        json.dump(identifiers, json_file)


def load_identifiers(path: Union[str, Path]) -> dict:
    with open(Path(path).with_suffix('.pkl'), 'rb') as f:
        return pickle.load(f)


def load_identifiers_json(path: Union[str, Path]) -> dict:
    with open(Path(path).with_suffix('.json'), 'r') as f:
        return json.load(f)


def check_ref_correspondence(raw_seqs, seq_dict, path):
    data = load_identifiers_json(path)
    for raw_id, seq in raw_seqs.items():
        ref_num = data[raw_id]
        if seq_dict[ref_num] != seq:
            print(f'Discrepancy for {raw_id}!')
    print('Reference Correspondence Check finished')


def uniprot_api_fetch_ensembl(embl_ids: Union[list, set]) -> dict:
    """
    :param embl_ids:
    :return:
    """
    query_url = 'https://www.uniprot.org/uniprot/?query={}+organism%3AHuman&sort=score&format=tab&columns=id'
    entry_url = 'https://www.uniprot.org/uniprot/{}.fasta'

    embl_records = dict()
    not_found = list()
    # url_vs_fasta = list()

    for _id in tqdm(sorted({_id.split('.')[0] for _id in embl_ids})):
        r = requests.get(query_url.format(_id))
        if not r.ok or not r.text:
            not_found.append(_id)
            continue

        found_id = r.text.strip().split('\n')[1]
        r = requests.get(entry_url.format(found_id))

        if not r.ok or not r.text:
            not_found.append(_id)
            continue

        seq = ''.join(r.text.strip().split('\n')[1:])
        embl_records[_id] = {'id': found_id, 'seq': seq, 'query': _id}
    if not_found:
        print(f'missing {len(not_found)}:')
        print(not_found)
    return embl_records
