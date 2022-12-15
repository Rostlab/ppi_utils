import concurrent.futures
import warnings
from pathlib import Path
from typing import Union, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from matplotlib.figure import Figure
from scipy import stats
from tqdm import tqdm

from ppi_utils.cfg import Config, CorrelationType
from ppi_utils.general import get_seq_hash

mpl.rcParams['figure.dpi'] = 200


def make_c_classes(test_tsv: Union[str, Path],
                   c3_fasta: Union[str, Path],
                   test_fasta: Union[str, Path]
                   ) -> pd.DataFrame:
    assert Path(c3_fasta).is_file(), f'C3 FASTA {c3_fasta.name} is missing!'
    c3_ids, test_fasta_ids = [{r.id for r in SeqIO.parse(
        f, 'fasta')} for f in [c3_fasta, test_fasta]]
    assert test_fasta_ids, 'WTF?'
    if not c3_ids:
        warnings.warn('No non-redundant sequences left; '
                      'cannot construct C3 test!', RuntimeWarning)
    assert not c3_ids - test_fasta_ids, \
        f'Forgot to replace {test_fasta} after "shrink_files_both_ways", ' \
        f'before the rostclust uniqueprot2d run?'
    if not test_fasta_ids - c3_ids:
        warnings.warn('Test set completely non-redundant, ' \
                      'cannot construct C1-2 sets!', RuntimeWarning)

    ppis = pd.read_csv(test_tsv, sep='\t', header=0)
    ppi_ids = set(np.unique(ppis.iloc[:, [0, 1]]))
    assert ppi_ids == test_fasta_ids, \
        f'The IDs in {test_fasta} and {test_tsv} should be the same! ' \
        f'Did you forget "shrink_files_both_ways" after the redundancy reduction?'

    _c123 = lambda df: 1 + df.iloc[:, 0].isin(c3_ids) + df.iloc[:, 1].isin(c3_ids)
    ppis['cclass'] = _c123(ppis)
    return ppis


def make_negatives(ppis: pd.DataFrame,
                   cfg: Config,
                   proteome: dict[int, dict[str, str]] = None,
                   quiet: bool = False,
                   ) -> tuple[pd.DataFrame, pd.DataFrame,
                              pd.DataFrame, Union[Figure, None]]:
    negatives, bias, fig, _ = find_negative_pairs(ppis, cfg, proteome, quiet)
    ppis['label'] = 1
    ppis = ppis[[c for c in ppis.columns if c not in
                 ('minlen', 'maxlen', 'degree_0', 'degree_1', 'n_seqs')]]
    sp = 9606 if 'species' not in ppis.columns else ppis.species.unique()[0]
    if 'species' not in negatives.columns:
        negatives['species'] = sp
    negatives['label'] = 0
    negatives.columns = ['hash_A', 'hash_B'] + list(negatives.columns[2:])
    if type(bias) == np.ndarray:
        bias = pd.DataFrame(bias.T, columns=['species', 'bias']
                            ).convert_dtypes()
    elif type(bias) == pd.DataFrame:
        bias = bias.T.reset_index().convert_dtypes()
        bias.columns = ['species', 'bias']
    return ppis, negatives, bias, fig


def get_train_lookup(dd: pd.DataFrame) -> dict[int, dict[str, int]]:
    assert dd.shape[1] == 3
    dd = dd.copy()
    dd.columns = ['ida', 'idb', 'label']
    dd.label = dd.label.astype(int)
    return {j: dict(zip(*np.unique(dd.loc[dd.label == j, ['ida', 'idb']],
                                   return_counts=True))) for j in [0, 1]}


def train_counts(dk: pd.DataFrame,
                 pairs: dict[int, dict[str, int]]) -> pd.Series:
    """
    :param dk: a DataFrame with the first two columns filled with IDs
    :param pairs: a dictionary, the result of get_train_lookup(dk)
    :return:
    """
    min_pos = dk.apply(lambda s: int(
        min(pairs[1].get(s[0], 0),
            pairs[1].get(s[1], 0))), axis=1)
    max_occ = dk.apply(lambda s: int(
        max(pairs[0].get(s[0], 0) + pairs[1].get(s[0], 0),
            pairs[0].get(s[1], 0) + pairs[1].get(s[1], 0)
            )), axis=1)
    return (min_pos / max_occ)  # .clip(upper=1)


def fetch_ratios(pairs: pd.DataFrame) -> pd.DataFrame:
    degrees = list()
    for l, df in pairs.groupby(list(pairs.columns[2:])):
        proteins, counts = np.unique(df.iloc[:, [0, 1]], return_counts=True)
        degrees.append([*l, np.vstack((np.arange(len(counts)),
                                       np.sort(counts)[::-1])).T])
    dt = pd.DataFrame(degrees, columns=list(
        pairs.columns[2:]) + ['degree'])
    dt = dt.explode('degree')
    dt[['x', 'degree']] = dt.degree.tolist()
    return dt


def fetch_degrees(pairs: pd.DataFrame, as_dict: bool = False
                  ) -> Union[pd.DataFrame, dict[str, int]]:
    pairs = (pairs[list(pairs.columns[:2]) + ['species']]
             .melt(id_vars='species', value_name='crc_hash')
             .iloc[:, [0, 2]].value_counts()
             .reset_index().rename(columns={0: 'degree'})
             )
    if not as_dict:
        pairs['species'] = pd.Categorical(pairs.species)
        return pairs
    else:
        return pairs.iloc[:, [1, 2]].set_index('crc_hash').to_dict()['degree']


def fetch_degree_frequencies(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = (fetch_degrees(pairs).iloc[:, [0, 2]]
             .value_counts().reset_index()
             .rename(columns={0: 'frequency'}))
    pairs['species'] = pd.Categorical(pairs.species)
    return pairs


def fetch_n_proteins(pairs: pd.DataFrame) -> pd.DataFrame:
    return fetch_degrees(pairs).species.value_counts()


def count_homodimers(pairs: pd.DataFrame) -> tuple[int, float, int]:
    homod = len(pairs.loc[pairs.hash_A == pairs.hash_B])
    return homod, np.round(homod / len(pairs), 4), len(pairs)


def make_validation_species(pairs: pd.DataFrame, species: set[int]
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pairs.loc[pairs.species.isin(species)]
    pairs = pairs.loc[~pairs.species.isin(species)].copy()
    train_ids = set(np.unique(pairs.iloc[:, [0, 1]]))
    df = df.loc[(~df.hash_A.isin(train_ids)) & (~df.hash_B.isin(train_ids))]
    return pairs, df


def make_validation_split(pairs: pd.DataFrame,
                          val_set_size: float = .5, seed: int = 42,
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed=seed)
    i, j = 0, 0
    val_proteins = set()
    val_ppis = pd.DataFrame()
    # iteratively add more proteins until we fulfill the target size
    while len(val_ppis) < val_set_size * 3 * len(pairs):
        i += 1
        val_ppis = pairs.loc[(pairs.hash_A.isin(val_proteins)) |
                             (pairs.hash_B.isin(val_proteins))]
        if j == len(val_ppis):
            # no new PPIs added: only homodimers in last step, or species barrier
            idx = rng.choice(range(len(pairs)))
            val_proteins |= set(pairs.iloc[idx, [0, 1]])
        else:
            # add another random protein
            val_proteins |= set(np.unique(val_ppis.iloc[:, [0, 1]]))
            j = len(val_ppis)
    print(f'{i} loops')

    idcs = sorted(set(val_ppis.index))
    # idcs = list(val_ppis.index)[:int(val_set_size * len(pairs))]
    # idcs = np.sort(rng.choice(
    #     list(val_ppis.index), size=int(len(pairs) * val_set_size), replace=False))
    n_idcs = np.delete(np.arange(0, len(pairs)), idcs)

    _train, _val = pairs.iloc[n_idcs, :].copy(), pairs.iloc[idcs, :].copy()

    # drop proteins that occur in train from val
    train_proteins = set(np.unique(_train.iloc[:, [0, 1]]))
    _val = _val.loc[~(_val.hash_A.isin(train_proteins))
                    & ~(_val.hash_B.isin(train_proteins))]

    return _train, _val


def find_negative_pairs_tuple(args):
    return find_negative_pairs(*args)


def find_negative_pairs(true_ppis: pd.DataFrame, cfg: Config,
                        proteome: dict[Union[str, int], dict[str, str]] = None,
                        quiet: bool = False, plot: bool = True,
                        shutup: bool = False,
                        ) -> tuple[pd.DataFrame, Union[float, np.ndarray],
                                   Union[Figure, None], int]:
    if proteome is None:
        proteome = dict()
    # recursive call for multi-species case
    if 'species' in true_ppis.columns and len(set(true_ppis.species)) > 1:
        if not quiet and not shutup:
            print(f'sampling negatives per species! aim for '
                  f'{int(len(true_ppis) * cfg.ratio)}')
        negatives, biases = list(), dict()
        tuples = list()
        for sp, ppis in true_ppis.groupby('species'):
            sp = int(sp) if str(sp).isnumeric() else str(sp)
            tuples.append((ppis, cfg, {sp: proteome.get(sp, dict())}, True, False, shutup))
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(tuples)) as executor:
            for n, b, f, s in executor.map(find_negative_pairs_tuple, tuples):
                n['species'] = s
                negatives.append(n)
                biases[s] = b
        # for n, b, f, s in process_map(find_negative_pairs_tuple, tuples,
        #                               max_workers=len(set(true_ppis.species))):
        #     n['species'] = s
        #     negatives.append(n)
        #     biases[s] = b
        negatives = pd.concat(negatives)
        bias = pd.DataFrame.from_dict(biases.items()).set_index(0).T
        if not shutup:
            print(f'{len(negatives)} negatives with overall '
                  f'{estimate_bias(true_ppis, negatives)[0]:.3f} '
                  f'and average per-species bias of '
                  f'{np.nanmean(bias):.3f}±{np.nanstd(bias):.3f} (std)')
        return negatives, bias, None, 0

    if 'species' in true_ppis.columns:
        sp = str(set(true_ppis.species).pop())
        sp = int(sp) if sp.isnumeric() else sp
        disable = False
    else:
        quiet = True
        disable = True
        sp = 0
    if shutup:
        quiet = True
        disable = True

    # map protein IDs to their sorting index
    uniq_true = {_id: idx for idx, _id in enumerate(
        np.unique(true_ppis.iloc[:, [0, 1]]))}
    uniq_neg = {idx: _id for _id, idx in uniq_true.items()}
    forward = np.vectorize(uniq_true.get)
    reverse = np.vectorize(uniq_neg.get)
    idx_ppis = forward(true_ppis.iloc[:, [0, 1]])

    # np.unique returns sorted values, so this works out
    proteins, counts = np.unique(true_ppis.iloc[:, [0, 1]], return_counts=True)
    n = len(proteins)

    rng = np.random.default_rng(seed=cfg.seed + (sp if type(sp) == int else 0))

    wants = np.floor(counts * cfg.ratio).astype(int)
    if cfg.strategy.value != 1:
        fl = np.floor(sum(counts) * cfg.ratio / n)
        wants = np.full_like(wants, fl).astype(int)
        if not disable and not quiet:
            tqdm.write(f'{sp}: {fl} is shared lower target')

    # make sure that `wants` is an integer vector and its sum as close to the target as possible
    idcs = list(rng.choice(n, size=n, replace=True,
                           p=(counts / sum(counts)) if cfg.strategy.value == 1 else None))
    while np.round(sum(counts) * cfg.ratio) > sum(wants):
        idx = idcs.pop(0)
        wants[idx] += 1

    ideal_neg_med = np.median(wants) + np.median(counts)
    wants = np.append(wants, 0)
    limit = sum(wants)

    if not quiet:
        tqdm.write(f'{sp}: {len(idx_ppis)} positives, aim for {limit // 2} negatives')

    # initialize the matrix
    mat = np.zeros((n + 1, n + 1), dtype=int)
    if not cfg.accept_homodimers:
        np.fill_diagonal(mat, 1)
        mat[-1, -1] = 0
    mat[idx_ppis[:, 0], idx_ppis[:, 1]] = 1
    mat[idx_ppis[:, 1], idx_ppis[:, 0]] = 1

    with tqdm(total=limit // 2, position=0, desc=str(sp), disable=disable) as pbar:
        while np.sum(wants[:n]):
            x = rng.choice(n, size=1, replace=False, p=wants[:n] / sum(wants[:n]))[0]
            wants[x] -= 1
            wants[-1] = max(0, 2 * wants[x] - (mat[x, :n] == 0) @ wants[:n])
            p_proteome = np.append((mat[x, :n] == 0) * wants[:n], wants[n])
            if not np.sum(p_proteome):
                p_proteome[-1] = 1
            y = rng.choice(n + 1, size=1, p=p_proteome / sum(p_proteome))[0]
            mat[y, x] -= 1  # tolerant against y=n
            if x != y:
                mat[x, y] -= 1
            wants[y] -= 1  # y=n will ignore this
            pbar.update(1)

    if (not quiet) or plot:
        fig, ax = plt.subplots()
        cmap = mpl.colors.ListedColormap(-(np.min(mat) + 1) * ['#6B0E30']
                                         + ['#D81B60', '#FFFFFF', '#1E88E5'])
        heat = sns.heatmap(mat,  # annot=True, linewidth=.2,
                           ax=ax, cmap=cmap, cbar=False)
        ax.set(box_aspect=1, xticks=[], yticks=[])
    else:
        fig = None

    # extract negative edges from the upper triangle
    negs = np.vstack(np.nonzero(np.triu(mat, k=0) < 0)).T
    # filter out the last column
    negs = negs[(negs[:, 0] < n) & (negs[:, 1] < n)]
    # convert back to crc64 hashes
    negatives = pd.DataFrame(reverse(negs)) if len(negs) else pd.DataFrame()
    if (not quiet or not len(negatives)) and not shutup:
        tqdm.write(f'{sp}: {len(negatives)}/{limit // 2} interactome negatives')

    # look up the indices of potential extra/proteome negatives
    idcs = np.flatnonzero(mat[:, n])
    extra = np.vstack((idcs, -mat[idcs, n])).T
    if len(extra) and len(proteome.get(sp, set())):
        negatives = pd.concat((negatives, find_proteome_negative_pairs(
            extra, set(proteome[sp]) - set(uniq_true), sp, cfg.seed, reverse,
            np.mean(counts) if hasattr(cfg, 'legacy') and cfg.legacy else ideal_neg_med,
            not quiet, False)[0]))
    if not len(negatives):
        if not quiet:
            print(f'{sp}: got 0 negatives!')
        return negatives, np.NaN, fig, sp

    bias = estimate_bias(true_ppis, negatives)[0]
    if not quiet:
        print(f'{sp}: got {len(negatives)} negatives with bias {bias:.3f}')
    return negatives, bias, fig, sp


def find_proteome_negative_pairs(extra: np.ndarray,
                                 proteome_ids: set[str],
                                 sp: Union[int, str], seed: int,
                                 idx_to_crc: Callable,
                                 ideal_median: float,
                                 verbose: bool = True,
                                 quiet: bool = False,
                                 ) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed=seed + 1 + (sp if type(sp) == int else 0))
    min_extra = np.max(extra[:, 1])
    extra_interactions = np.sum(extra[:, 1])
    avg_extra = np.ceil(extra_interactions / ideal_median).astype(int)
    n_extra = min(len(proteome_ids), max(min_extra, avg_extra))
    if verbose:
        print(f'{sp}: need {min_extra} extra proteins for {len(extra)} hubs; '
              f'select {n_extra} from {len(proteome_ids)} additional proteins. '
              f'Try to create {extra_interactions} interactions, '
              f'ideally {ideal_median} per protein.')

    extra_proteins = list(rng.choice(sorted(proteome_ids), size=n_extra, replace=False))
    if not quiet:
        tqdm.write(f'{sp} extras hash: {get_seq_hash(":".join(extra_proteins))[4:]}')

    extra_pairs = list()
    for p_idx, n_partners in extra:
        # for each protein missing negatives, try to find as many as necessary
        p = idx_to_crc(p_idx)
        partners = rng.choice(extra_proteins, size=min(
            n_partners, len(extra_proteins)), replace=False)
        # the new, *proteome* interaction partner is always in the second column
        extra_pairs.extend([(p, partner) for partner in partners])

    extras = pd.DataFrame(extra_pairs).astype(str)
    if verbose:
        print('proteome interactions:')
        print(pd.DataFrame(np.unique(extras.iloc[:, 1],
                                     return_counts=True)[1])
              .describe().round(2).T)
    # extra_crcs = set(extras.iloc[:, 1])
    return extras, extra_interactions - len(extras)


def estimate_bias_per_species(pairs: pd.DataFrame,
                              corrtype: CorrelationType = CorrelationType.PEARSON,
                              ) -> pd.DataFrame:
    bias = [(sp, estimate_bias(spdf, corrtype=corrtype)[0])
            for sp, spdf in pairs.groupby('species')]
    return pd.DataFrame.from_records(bias, columns=['species', 'bias'])


def estimate_bias(positives: pd.DataFrame,
                  negatives: pd.DataFrame = None,
                  corrtype: CorrelationType = CorrelationType.PEARSON,
                  ) -> tuple[float, float]:
    """
    Calculate the similarity between two sets of protein pairs:
    the Spearman or Pearson correlation coefficient between their
    protein-appearance frequency vectors.
    """
    if negatives is None:
        positives, negatives = sep_plus_minus(positives)
    plus, minus = [dict(zip(*np.unique(ar.iloc[:, [0, 1]], return_counts=True)))
                   for ar in (positives, negatives)]

    for p, m in ((plus, minus), (minus, plus)):
        m.update({k: 0 for k in p.keys() - m.keys()})
    if corrtype.value == 0:
        return stats.pearsonr(*[[ar[k] for k in sorted(ar.keys())]
                                for ar in (plus, minus)])
    elif corrtype.value == 1:
        return stats.spearmanr(*[[ar[k] for k in sorted(ar.keys())]
                                 for ar in (plus, minus)], axis=1)
    else:
        assert False, 'illegal correlation type'


def sep_plus_minus(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert 'label' in pairs.columns
    negatives = pairs.loc[pairs.label == 0].copy()
    positives = pairs.loc[pairs.label == 1].copy()
    assert len(positives), 'no positives in passed DataFrame'
    assert len(negatives), 'no negatives in passed DataFrame'
    return positives, negatives


def find_multi_species_ppis(ppi_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df for i, df in ppi_df.groupby(
        ['UniprotID_A', 'UniprotID_B']) if len(df) > 1]
                     + [ppi_df.loc[ppi_df.species == 'is there a marsupilami?']])


def find_multi_species_proteins(ppis: pd.DataFrame) -> pd.DataFrame:
    dfs = list()
    for c in 'AB':
        pairs = ppis[[f'UniprotID_{c}', 'species']].drop_duplicates()
        dfs += [df for i, df in pairs.groupby(f'UniprotID_{c}') if len(df) > 1]
        dfs.append(pairs.loc[pairs.species == 'is there a marsupilami?'].values)
    return pd.DataFrame(np.vstack(dfs), columns=['UniprotID', 'species']) \
        .drop_duplicates().groupby('UniprotID') \
        .agg({'species': lambda l: sorted(l.astype(int).tolist())}).reset_index()


def ppis_to_sp_lookup(ppis: pd.DataFrame) -> dict[str, int]:
    return (ppis[['hash_A', 'hash_B', 'species']]
    .melt(id_vars='species')[['species', 'value']]
    .set_index('value').to_dict()['species'])
