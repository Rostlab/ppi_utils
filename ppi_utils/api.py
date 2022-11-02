import json
import re
import shutil
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Union, Iterable

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dataclass_wizard import JSONWizard
from tqdm import tqdm

from ppi_utils.general import get_seq_hash, to_fasta, file_hash


@dataclass
class Organism(JSONWizard):
    taxon_id: int
    scientific_name: str = ''
    common_name: str = ''


@dataclass
class UniParcCrossReference(JSONWizard):
    database: str
    id: str
    version_i: int
    active: bool
    created: str
    last_updated: str
    organism: Organism = None
    version: int = -1
    gene_name: str = ''
    protein_name: str = ''


@dataclass
class UpiSequence(JSONWizard):
    value: str
    length: int
    crc64: str
    md5: str
    mol_weight: int = 0


@dataclass
class InterproGroup(JSONWizard):
    id: str
    name: str


@dataclass
class Location(JSONWizard):
    start: int
    end: int


@dataclass
class SequenceFeatures(JSONWizard):
    interpro_group: InterproGroup
    database: str
    database_id: str
    locations: list[Location]


@dataclass
class UpiEntry(JSONWizard):
    uni_parc_id: str
    uni_parc_cross_references: list[UniParcCrossReference]
    sequence: UpiSequence
    oldest_cross_ref_created: str
    most_recent_cross_ref_updated: str
    sequence_features: list[SequenceFeatures] = None
    references_original_id: bool = False
    active_uniprot_ids: dict[str, Organism] = field(default_factory=dict)

    def check_against(self, old_id: str) -> None:
        """Check if this entry references this UniProtID as inactive and
        has an active one; sort and filter its cross-references; keep track
        of active IDs and the species they map to."""
        self.uni_parc_cross_references.sort(key=lambda cross_ref: cross_ref.id)
        crs = list()
        for cr in self.uni_parc_cross_references:
            if not cr.active and cr.id == old_id:
                self.references_original_id = True
            elif cr.active and cr.database.startswith('UniProtKB'):
                self.active_uniprot_ids[cr.id] = cr.organism
                crs.append(cr)
        self.uni_parc_cross_references = crs
        if self.references_original_id and self.active_uniprot_ids:
            assert len(self.uni_parc_cross_references)


def uniprot_api_fetch(uniprot_ids: Union[set, list],
                      out_file: Union[str, Path] = Path('uniprot'),
                      known_species: Iterable[int] = None
                      ) -> Union[str, dict[str, str]]:
    """
    Download sequences from UniProt: First, post the given set of
    UniProt IDs to the ID mapping site and retrieve `tab` and `fasta`.
    Then, check UniParc for up-to-date replacement entries, finally
    use obsolete versions of the given IDs.
    :param uniprot_ids:
    :param out_file:
    :param known_species:
    :return:
    """
    uniprot_ids = set(uniprot_ids)
    out_file = Path(out_file)
    if out_file.is_dir():
        out_file = out_file / 'uniprot'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    server = 'https://rest.uniprot.org'
    idpost = '/idmapping/run'
    idstatus = '/idmapping/status/'
    idget = '/idmapping/stream/'
    idfasta = '/idmapping/uniprotkb/results/stream/{}?format=fasta'
    rest_url = '/uniprotkb/{}.fasta'
    archive_url = '/unisave/{}?format=fasta&uniqueSequences=true'
    txt_url = '/unisave/{}?format=txt'

    params = {
        'ids': ','.join(sorted(uniprot_ids)),
        'from': 'UniProtKB_AC-ID',
        'to': 'UniProtKB',
    }
    errors = list()
    seen_crcs = set()
    seen_descs = set()

    response = requests.post(server + idpost, params)
    if response.status_code != 200:
        return response.content.decode('utf-8')
    job_id = response.json()['jobId']
    print(f'jobId: {job_id}')

    response = requests.get(server + idstatus + job_id)
    r_json = response.json()
    while r_json.get('jobStatus', 'nope').upper() == 'RUNNING':
        print('.', end='')
        sleep(10)

    print(f'{out_file.stem}: query + tab ...', end='')
    response = requests.get(server + idget + job_id)
    r_json = response.json()
    from_to = pd.DataFrame(r_json.get('results', list()),
                           columns=['from', 'to']).rename(columns={'from': 'From', 'to': 'To'})
    forgot = r_json.get('failedIds', list())
    print(f' {len(from_to)}:{len(forgot)}')
    from_to.to_csv(out_file.with_suffix('.tab'), sep='\t', index=False)

    print(f'{out_file.stem}: fasta ...', end='')
    with requests.get(server + idfasta.format(job_id)) as response, \
            open(out_file.with_suffix('.fasta'), 'w') as fasta:
        fasta.write(response.text)
    print('API FASTA:', file_hash(out_file.with_suffix('.fasta')))

    # save intermediary results in a dict, not via pd.loc
    m_dict = dict()

    shutil.move(out_file.with_suffix('.fasta'), out_file.with_suffix('.fasta.bak'))
    with out_file.with_suffix('.fasta').open('w') as seq_fasta, \
            out_file.with_suffix('.hash.fasta').open('w') as hash_fasta, \
            out_file.with_suffix('.fasta.bak').open('r') as in_fasta:
        for r in tqdm(SeqIO.parse(in_fasta, 'fasta'), desc='hash FASTA'):
            _id = r.id.split('|')[1]
            crc = get_seq_hash(r.seq)
            m_dict[_id] = [crc, 'query', r.description]

            if r.description in seen_descs:
                assert crc in seen_crcs
                continue
            seen_descs.add(r.description)
            to_fasta(r.description, r.seq, seq_fasta)

            if crc in seen_crcs:
                continue
            seen_crcs.add(crc)
            to_fasta(crc, r.seq, hash_fasta)
    out_file.with_suffix('.fasta.bak').unlink(missing_ok=True)

    from_to = pd.concat([from_to, pd.DataFrame(
        zip(forgot, forgot), columns=['From', 'To'])])
    from_to['crc_hash'] = ''

    def _refresh_and_save_from_to_():
        from_to.loc[from_to.crc_hash == '', ['crc_hash', 'source', 'description']] = \
            from_to.loc[from_to.crc_hash == '', 'To'].apply(
                lambda t: m_dict.get(t, ['', '', ''])).to_list()

        from_to.sort_values(by='crc_hash').to_csv(
            out_file.with_suffix('.tsv'), index=False, sep='\t')

    _refresh_and_save_from_to_()

    is_known_id = lambda _id: _id in m_dict.keys()
    is_known_sp = lambda sp: bool(sp in known_species) if known_species is not None else False

    def cache(text, old_uniprot, new_uniprot, source_db, hash_file, clear_file,
              ox: int = None, os: str = ''):
        _desc, *seq = text.strip().lstrip('>').split('>')[0].split('\n')
        if ox is not None and 'OX=' not in _desc:
            _desc += f' OX={ox}'
        if os and 'OS=' not in _desc:
            _desc += f' OS={os}'
        seq = ''.join(seq)
        crc = get_seq_hash(seq)
        m_dict[old_uniprot] = [crc, source_db, _desc]
        # from_to.loc[from_to.To == old_uniprot, 'To'] = new_uniprot

        if crc in seen_crcs:
            return
        seen_crcs.add(crc)
        to_fasta(crc, seq, hash_file)
        to_fasta(_desc, seq, clear_file)

    if uniparcs := (set(from_to.loc[from_to.crc_hash == '', 'To']) - m_dict.keys()):
        with out_file.with_suffix('.hash.fasta').open('a') as hash_fasta, \
                out_file.with_suffix('.fasta').open('a') as seq_fasta:
            # print(' '.join(sorted(uniparcs)))
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            for _id in tqdm(sorted(uniparcs), desc='fetch UniParc'):
                ext = f'/uniparc/search?query={_id}'
                r = requests.get(server + ext, headers=headers)
                if not r.ok:
                    r.raise_for_status()
                    continue
                upi_entries = [UpiEntry.from_dict(d) for d in r.json()['results']]

                # filter and sort the entries
                ok_entries = list()
                for entry in upi_entries:
                    entry.check_against(_id)
                    if entry.references_original_id and entry.active_uniprot_ids:
                        ok_entries.append(entry)
                ok_entries.sort(key=lambda en: en.uni_parc_id)

                found = False
                # look for known UniProtIDs
                for entry in ok_entries:
                    for new_id, organism in entry.active_uniprot_ids.items():
                        if is_known_id(new_id):
                            r = requests.get(server + rest_url.format(new_id))
                            if not r.ok:
                                errors.append(f'uniparc {_id} known ID: {new_id}')
                                continue
                            cache(r.text, _id, new_id, 'uniparc', hash_fasta, seq_fasta,
                                  ox=organism.taxon_id, os=organism.scientific_name)
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

                # look for known species
                for entry in ok_entries:
                    for new_id, organism in entry.active_uniprot_ids.items():
                        if is_known_sp(organism.taxon_id):
                            r = requests.get(server + rest_url.format(new_id))
                            if not r.ok:
                                errors.append(f'uniparc {_id} ID {new_id} to known '
                                              f'sp: {organism.taxon_id}')
                                continue
                            cache(r.text, _id, new_id, 'uniparc', hash_fasta, seq_fasta,
                                  ox=organism.taxon_id, os=organism.scientific_name)
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

                # use any remaining active UniProtID
                for entry in ok_entries:
                    for new_id, organism in entry.active_uniprot_ids.items():
                        r = requests.get(server + rest_url.format(new_id))
                        if not r.ok:
                            errors.append(f'uniparc {_id} ID {new_id} as new')
                            continue
                        cache(r.text, _id, new_id, 'uniparc', hash_fasta, seq_fasta,
                              ox=organism.taxon_id, os=organism.scientific_name)
                        found = True
                        break
                    if found:
                        break
        _refresh_and_save_from_to_()

    if isoforms := (set(from_to.loc[from_to.crc_hash == '', 'To']) - m_dict.keys()):
        with out_file.with_suffix('.hash.fasta').open('a') as hash_fasta, \
                out_file.with_suffix('.fasta').open('a') as seq_fasta:
            # print(' '.join(sorted(isoforms)))
            for _id in tqdm(sorted(isoforms), desc='fetch isoforms/archive'):
                source = 'isoform' if '-' in _id else 'entry'
                os, ox = None, None
                r = requests.get(server + rest_url.format(_id))
                if not r.ok:
                    errors.append(f'{source}: {_id}')
                    continue
                if not r.text:
                    r = requests.get(server + archive_url.format(_id))
                    source += '-archive'
                    r2 = requests.get(server + txt_url.format(_id))
                    for line in r2.text.strip().split('\n'):
                        if line[:2] == 'OS':
                            m = re.match('OS\s+(?P<species>[^\.$]+)[\.$]', line)
                            assert m, f'No match: {line}'
                            os = m.groupdict()['species']
                        if line[:2] == 'OX':
                            m = re.match('OX\s+.*NCBI_TaxID=(?P<taxid>\d+)', line)
                            assert m, f'No match: {line}'
                            ox = int(m.groupdict()['taxid'])
                            break
                cache(r.text.strip().lstrip('>').split('>')[0], _id, _id, source,
                      hash_fasta, seq_fasta, ox=ox, os=os)
        _refresh_and_save_from_to_()

    if errors:
        print('\n'.join(errors))
        for e in errors:
            m_dict[e.split(':')[-1].strip()] = ['', '']

    simple_from_to = from_to.iloc[:, [0, 2]].copy().set_index('From').to_dict()['crc_hash']
    with open(out_file.with_suffix('.json'), 'w') as json_file:
        json.dump(simple_from_to, json_file)
    missing = sorted(set(from_to.To) - m_dict.keys())
    if missing:
        print('missing:\n' + ' '.join(missing))

    return simple_from_to


def fetch_huri_seqs(huri_ids: dict,
                    out_file: Union[str, Path] = Path('huri'),
                    ) -> tuple[dict[str, SeqRecord], dict[str, str]]:
    if (fasta_file := out_file.with_suffix('.hash.fasta')).is_file():
        print(f'loading from {fasta_file} and {out_file.with_suffix(".json")}')
        fasta = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))
        with out_file.with_suffix('.json').open('r') as json_file:
            from_to = json.load(json_file)
        return fasta, from_to

    fasta, from_to = ensembl_api_fetch(huri_ids['ensembl'],
                                       out_file.with_name('ensembl'))
    uniprot_from_to = uniprot_api_fetch(huri_ids['uniprotkb'],
                                        out_file.with_name('uniprot'))
    fasta.update(SeqIO.to_dict(SeqIO.parse(
        out_file.with_name('uniprot.hash.fasta'), 'fasta')))
    SeqIO.write(fasta.values(), out_file.with_suffix('.fasta'), 'fasta')
    from_to.update(uniprot_from_to)
    with out_file.with_suffix('.json').open('w') as json_file:
        json.dump(from_to, json_file)

    return fasta, from_to


def fetch_taxonomic_info(taxonomic_ids: pd.Series) -> pd.DataFrame:
    server = 'https://rest.ensembl.org'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    common_names = dict()
    for tax_id in tqdm(taxonomic_ids.unique()):
        ext = f'/taxonomy/id/{tax_id}?'
        r = requests.get(server + ext, headers=headers)
        if not r.ok:
            r.raise_for_status()
            continue
        decoded = r.json()
        common_names[tax_id] = decoded['name']
    return pd.DataFrame.from_dict(common_names, orient='index', columns=['name']) \
        .rename_axis('species').reset_index() \
        .sort_values(by='species').join(
        taxonomic_ids.value_counts(), on='species', rsuffix='n_ppis'
    ).reset_index(drop=True).rename(columns=dict(speciesn_ppis='n_ppis'))


def ensembl_api_fetch(embl_ids: Union[set, list],
                      out_file: Union[str, Path] = Path('ensembl'),
                      batch_size: int = 50,
                      ) -> tuple[dict[str, SeqRecord], dict[str, str]]:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    batch_size = min(batch_size, 50)
    server = 'https://rest.ensembl.org'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    embl_records = dict()
    for archive in [False, True]:
        # cut off version identifiers because we can't use them for queries anyway
        embl_ids = sorted({_id.split('.')[0] for _id in embl_ids} - embl_records.keys())
        if not embl_ids:
            continue
        ext = f'/{"archive" if archive else "sequence"}/id?type=protein;'
        _s = '"id"' if archive else '"ids"'
        for j in tqdm(range(0, batch_size * int(
                np.ceil(len(embl_ids) / batch_size)), batch_size),
                      desc=('archive ' if archive else 'Ensembl ') + 'batch'):
            l = embl_ids[j:j + batch_size]
            if not l:
                continue
            r = requests.post(server + ext, headers=headers,
                              data=('{ ' + _s + ': ' + str(l) + ' }').replace('\'', "\""))
            if not r.ok:
                continue
            for rj in r.json():
                if archive and rj['peptide'] is not None:
                    embl_records[rj['id']] = rj
                elif rj['seq'] is not None:
                    embl_records[rj['query']] = rj
            j += batch_size

    with out_file.with_suffix('.response.json').open('w') as json_file:
        json.dump(embl_records, json_file)

    embl_records = {_id: r.get('seq', r.get('peptide'))
                    for _id, r in embl_records.items()}

    hash_records = dict()
    from_to = dict()
    with out_file.with_suffix('.fasta').open('w') as clear_fasta, \
            out_file.with_suffix('.hash.fasta').open('w') as hash_fasta:
        for _id in sorted(embl_records.keys()):
            seq = embl_records[_id]
            to_fasta(_id, seq, clear_fasta)
            _hash = get_seq_hash(seq)
            hash_records[_hash] = SeqRecord(Seq(seq), id=_hash, description='')
            from_to[_id] = _hash
            to_fasta(_hash, seq, hash_fasta)
    with out_file.with_suffix('.json').open('w') as json_file:
        json.dump(from_to, json_file)

    missing = sorted(set(embl_ids) - embl_records.keys())
    if missing:
        print('missing:\n' + ' '.join(missing))

    return hash_records, from_to


def download_y2h_interactome(
        out_file: Union[str, Path] = Path('hi_union.psi'),
        target_hash: str = '25c9566fe06fe0682e4183b326fef7ff705'
                           'fbe4d52fdd276242b030ec746d87f') -> Path:
    # kudos to https://stackoverflow.com/a/37573701
    out_file = Path(out_file)
    if out_file.is_file():
        if file_hash(out_file) == target_hash:
            print('already downloaded and SHA256 checks out')
            return out_file
    url = 'http://www.interactome-atlas.org/data/HI-union.psi'
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 ^ 2  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes,
                        unit='iB', unit_scale=True,
                        position=0, leave=True)
    with out_file.open('wb') as file:
        for data in response.iter_content(block_size):
            _ = progress_bar.update(len(data))
            _ = file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print('ERROR, something went wrong')
    return out_file


def fetch_proteomes(species: set,
                    proteome_dir: Path = Path('proteomes')) -> None:
    url = 'https://rest.uniprot.org/proteomes/stream?' \
          'fields=upid,organism,organism_id,protein_count' \
          '&format=tsv&query=(proteome_type:1) AND ({})' \
        .format(' OR '.join(f'(organism_id:{sp})' for sp in species))

    r = requests.get(url)
    tab = pd.read_csv(StringIO(r.text), sep='\t')
    assert len(set(tab['Proteome Id'])) == len(tab), \
        'Got multiple/no proteomes for some species'

    if missing := species - set(tab['Organism Id']):
        if 37296 in missing:
            s = ['UP000097197', 'Human herpesvirus 8 (HHV-8)', 37296, 72]
            tab = pd.concat([tab, pd.DataFrame(s, index=tab.columns).T])
            missing.remove(37296)
        if 10600 in missing:
            s = ['UP000007676', 'Human papillomavirus type 6b', 10600, 9]
            tab = pd.concat([tab, pd.DataFrame(s, index=tab.columns).T])
            missing.remove(10600)
    if missing:
        raise ValueError(f'Missing proteomes: {" ".join((str(j) for j in missing))}')

    proteome_dir.mkdir(exist_ok=True, parents=True)
    proteome_url = 'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=((proteome:{}))'
    with tqdm(tab.iterrows(), total=len(tab)) as pbar:
        for i, proteome in pbar:
            pbar.set_postfix(batch=proteome['Organism'].split('(')[0].strip())
            r = requests.get(proteome_url.format(proteome['Proteome Id']), stream=True)
            with (proteome_dir / f'{proteome["Organism Id"]}.fasta').open('wb') as fd:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    fd.write(chunk)
