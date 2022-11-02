import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from Bio import SeqIO
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Union


@mpl.style.context('seaborn')
def filter_ppis_and_fasta_by_len(
        ppis: pd.DataFrame, fasta: dict,
        min_seq_len: int = 50, max_seq_len: int = 800
) -> tuple[pd.DataFrame, dict, Figure]:
    fasta_lens = {_id: len(seq) for _id, seq in fasta.items()}
    fetch_len = np.vectorize(fasta_lens.get)
    lens = fetch_len(ppis.iloc[:, [0, 1]])
    ppis[['minlen', 'maxlen']] = np.sort(lens, axis=1)

    loss_maxl = {maxl: len(ppis.loc[ppis.maxlen > maxl, 'maxlen'])
                       / len(ppis) for maxl in range(0, int(1e4), 25)}
    loss_minl = {minl: len(ppis.loc[ppis.maxlen < minl, 'minlen'])
                       / len(ppis) for minl in range(0, 400, 5)}

    loss_maxl = (pd.DataFrame.from_dict(
        loss_maxl, orient='index', columns=['loss'])
                 .reset_index().rename(columns=dict(index='len')))
    loss_minl = (pd.DataFrame.from_dict(
        loss_minl, orient='index', columns=['loss'])
                 .reset_index().rename(columns=dict(index='len')))

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharey=True, facecolor='None')

    ax = axes[0]
    sns.ecdfplot(fasta_lens.values(), legend=False, ax=ax)
    ax.set(xlabel='sequence length', ylabel='', xscale='log', box_aspect=1, xlim=(20, 1e4))
    ax.axvline(min_seq_len, lw=.5)
    ax.axvline(max_seq_len, lw=.5)

    ax = axes[1]
    sns.lineplot(data=loss_minl, x='len', y='loss', legend=False, ax=ax)
    ax.set(box_aspect=1, xlabel='lost PPIs over min len',
           ylabel='', xlim=(0, max(400, min_seq_len + 100)), ylim=(0, 1))
    ax.axvline(min_seq_len, lw=.5)

    ax = axes[2]
    sns.lineplot(data=loss_maxl, x='len', y='loss', legend=False, ax=ax)
    ax.set(box_aspect=1, xlabel='lost PPIs over max len',
           ylabel='', xlim=(min(500, max_seq_len - 100),
                            max(2000, max_seq_len + 100)), ylim=(0, 1))
    ax.axvline(max_seq_len, lw=.5)

    plt.tight_layout()

    k = len(ppis)
    ppis = ppis.loc[(ppis.minlen >= min_seq_len)
                    & (ppis.maxlen <= max_seq_len)].copy()
    l = len(ppis)
    if k > l:
        print(f'dropped {k - l}/{k} PPIs from length filtering')
    ppis, fasta = shrink_both_ways(ppis, fasta.copy())

    return ppis, fasta, fig


def dedup_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs.iloc[:, [0, 1]] = np.sort(pairs.iloc[:, [0, 1]], axis=1)
    return pairs.drop_duplicates()


def shrink_both_ways(pairs: pd.DataFrame, seqs: dict
                     ) -> tuple[pd.DataFrame, dict]:
    pair_ids = set(np.unique(pairs.iloc[:, [0, 1]]))
    seq_ids = set(seqs)
    # arithmetics
    n_pair_ids = len(pair_ids)
    n_pids = len(pair_ids - seq_ids)
    n_rows = len(pairs)
    # filter the pairs first
    pairs = pairs.loc[(pairs.iloc[:, 0].isin(seq_ids))
                      & (pairs.iloc[:, 1].isin(seq_ids))].copy()
    # there might be less pair_ids than seq_ids now
    pair_ids = set(np.unique(pairs.iloc[:, [0, 1]]))
    # arithmetics
    nn_rows = n_rows - len(pairs)
    n_sids = len(seq_ids - pair_ids)
    # therefore, drop whatever sequences don't have PPIs anymore
    for del_id in seq_ids - pair_ids:
        seqs.pop(del_id)
    if n_pids + nn_rows + n_sids:
        print(f'dropped {n_pids}/{n_pair_ids} table IDs '
              f'and {nn_rows}/{n_rows} rows, and {n_sids}/{len(seq_ids)} sequence IDs')
    return pairs, seqs


def shrink_files_both_ways(tsv: Union[str, Path], fasta: Union[str, Path],
                           new_tsv_name: Union[str, Path] = None,
                           new_fasta_name: Union[str, Path] = None,
                           ) -> None:
    tsv, fasta = Path(tsv), Path(fasta)
    pairs = pd.read_csv(tsv, sep='\t', header=0).astype(str)
    records = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))

    pairs, records = shrink_both_ways(pairs, records)

    if new_tsv_name and (t := Path(new_tsv_name)) != tsv:
        tsv = t
        tsv.parent.mkdir(parents=True, exist_ok=True)
    else:
        shutil.move(tsv, tsv.with_suffix('.tsv.bak'))
    if new_fasta_name and (p := Path(new_fasta_name)) != fasta:
        fasta = p
        fasta.parent.mkdir(parents=True, exist_ok=True)
    else:
        shutil.move(fasta, fasta.with_suffix('.fasta.bak'))

    pairs.to_csv(tsv, sep='\t', header=True, index=False)
    SeqIO.write((records[_id] for _id in sorted(records)), fasta, 'fasta')


def remove_ids_from(ppis: pd.DataFrame, black_list_fasta: Path) -> pd.DataFrame:
    ids = {r.id for r in SeqIO.parse(black_list_fasta, 'fasta')}
    return ppis.loc[~(ppis.hash_A.isin(ids)) & ~(ppis.hash_B.isin(ids))]


def drop_homodimers(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.loc[pairs.hash_A != pairs.hash_B]
