from seaborn import JointGrid
from typing import Union

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ppi_utils.pairs import fetch_ratios, fetch_degrees, \
    fetch_degree_frequencies, fetch_n_proteins, \
    count_homodimers, estimate_bias, estimate_bias_per_species, sep_plus_minus

mpl.rcParams['figure.dpi'] = 200
sns.reset_defaults()
sns.set_style({'figure.facecolor': 'None'})


@mpl.style.context('seaborn-poster')
def draw_toy_ppis(ppis: pd.DataFrame,
                  n_dict: dict[str, pd.DataFrame],
                  seed: int = 42,
                  extra_nodes: set = None
                  ) -> tuple[Figure, Figure]:
    if not extra_nodes:
        extra_nodes = set()

    def _edges_from_pd(df):
        return ((a, b) for _, a, b in df.iloc[:, [0, 1]].itertuples())

    def _minmax(ar, _min=60, _max=320):
        ar = np.array(ar)
        mi = np.min(ar)
        ma = np.max(ar)
        return (ar - mi) / (ma - mi) * (_max - _min) + _min

    # some magic so that the order of the plots stays
    # the same, but the true PPIs still come first
    keys = list(n_dict.keys())[::-1] + ['positive']
    n_dict['positive'] = ppis
    nodes = set(np.unique(ppis.iloc[:, [0, 1]])) | extra_nodes
    densest = 0, None
    graphs = dict()

    for key in reversed(keys):
        negatives = n_dict[key]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(_edges_from_pd(negatives), attr=dict(source=key))
        graphs[key] = G

        if (l := len(G.edges)) > densest[0]:
            densest = l, G

    # H = nx.Graph(densest[1])
    H = nx.Graph(graphs['positive'])
    H.add_edges_from(_edges_from_pd(ppis))
    pos = nx.spring_layout(H, iterations=100, seed=seed)
    pos = nx.kamada_kawai_layout(H)
    circ = nx.circular_layout(H)

    figs = list()
    for anno, lay in enumerate((pos, circ)):
        fig, axes = plt.subplots(1, len(n_dict))
        for ax, (key, G) in zip(axes, graphs.items()):
            nx.draw(H, pos=lay, ax=ax, alpha=.4, edge_color='#FF70A4',
                    with_labels=False, node_size=0)
            nx.draw(G, pos=lay, ax=ax, with_labels=True,
                    node_size=_minmax([G.degree[v] for v in G]),
                    node_color=[float(G.degree(v)) for v in G],
                    cmap=mpl.colors.ListedColormap(['#E53374', '#1E88E5', '#FFC107', '#00E0BB'][::-1]),
                    )
            if anno:
                ax.set(box_aspect=1)
            else:
                ax.set(box_aspect=1, ylim=(-1.2, None), xlim=(-1, None))
                bias = estimate_bias(ppis, n_dict[key])[0]
                ax.annotate(key + (f':\n{bias:.2f}' if key != 'positive' else ''),
                            xy=(-1, -1.2))
        fig.tight_layout()
        figs.append(fig)

    return figs[0], figs[1]


def plot_homodimer_share(pairs: pd.DataFrame, ratio: float = 10.0) -> Figure:
    df = pairs.groupby('species').apply(count_homodimers).reset_index()
    df[['_count', 'share', 'n_ppis']] = df[0].to_list()
    df['n_prots'] = df.species.apply(fetch_n_proteins(pairs).to_dict().get)

    df['_diag'] = df.n_prots.apply(lambda n: 2 / (n + 1))
    df['filled'] = df._count / df.n_prots
    a = 1 / (ratio + 1)
    df['enough'] = df.filled.apply(  # 'easy' if n < .5 * a else
        lambda n: 'yes' if n < a else 'close' if n < 2 * a else 'no')

    # np.asarray(['easy', 'yes', 'close', 'no'])[np.clip(np.asarray(
    # df.filled.values * (c.ratio  + 1)).astype(int), a_min=0, a_max=2)]

    fig, ax = plt.subplots(figsize=(8, 3), facecolor='None')
    scatter = sns.scatterplot(data=df,
                              x='n_ppis',
                              y='share',
                              hue='filled',
                              style='enough', style_order=['yes', 'close', 'no'],
                              s=40,
                              ax=ax,
                              )
    ax.set(xscale='log',  # ylim=(None, 1),
           xlabel='number of PPIs per species dataset',
           ylabel='homodimer share',
           )

    ax.legend(bbox_to_anchor=(1, 0), loc='lower left', frameon=False)
    # handles, labels = fig.axes[0].get_legend_handles_labels()
    # l1 = handles[:6], labels[:6]
    # l2 = handles[6:], labels[6:]
    # l1 = ax.legend(*l1, bbox_to_anchor=(1, 0), loc='lower left', frameon=False)
    # l2 = ax.legend(*l2, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    # fig.axes[0].add_artist(l1)

    # df = df.sort_values(by='n_ppis')
    for i, point in enumerate(df.itertuples()):
        # if i % 2:
        #     p = 1.05
        #     q = .03
        #     ha = 'left'
        #     va = 'bottom'
        # else:
        #     p = .95
        #     q = -.03
        #     ha = 'right'
        #     va = 'top'

        ax.annotate(point.species,
                    (point.n_ppis * 1.05,
                     point.share + .03),
                    rotation=50, size=6, zorder=0)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    return fig


def plot_theoretical_homodimer_share() -> Figure:
    with sns.plotting_context('poster'):
        # , mpl.rc_context({'figure.dpi': 64}):  # mpl.style.context('seaborn-poster')
        x = np.logspace(0, 4.5, 1000)
        y = 2 / (1 + x)
        fig, ax = plt.subplots(figsize=(6, 6))  # , facecolor='white')
        fg = sns.lineplot(x=x, y=y, ax=ax, color='#1E88E5', lw=4)
        # sns.lineplot(x=x, y=y/11, ax=ax, color='#D81B60', lw=2)
        # sns.lineplot(x=x, y=2/(x), ax=fg.ax, color='#D81B60')
        ax.set(xscale='log', xlabel='', ylabel='', box_aspect=1)  # , xticks=[1, 2, 3])
        ax.set_title('2 / (x + 1)', fontsize=32)
        # fg.set(xlim=(0, 100))
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
    return fig


def plot_interactome_sizes(taxonomy: pd.DataFrame,
                           val_species: set = None, min_x: int = None,
                           h: float = 6) -> Figure:
    tax = taxonomy.copy()
    tax['species'] = pd.Categorical(tax.species)
    order = list(tax.species)[::-1]
    if val_species is not None:
        tax['set_name'] = tax.species.apply(lambda sp: sp in val_species)
        pal = {1: '#D81B60', 0: '#1E88E5'}
    else:
        assert 'set_name' in tax
        pal = {'train': '#1E88E5', 'validate': '#D81B60',
               'test': '#FFC107', 'D-SCRIPT': '#004D40'}

    j = sns.catplot(
        data=tax,
        x='n_ppis',
        y='species',
        hue='set_name',
        palette=pal,
        order=order,
        s=7,
        height=len(set(tax.species)) / h,
        aspect=2,
        zorder=1000,
        legend=True
    )
    j.set(xscale='log', xlabel=None,
          ylim=(None, -.5), xlim=(min_x, max(tax.n_ppis) * 1.4))
    j.ax.get_yaxis().set_visible(False)
    j.despine(left=True, bottom=True)
    j.ax.xaxis.tick_top()
    j.ax.xaxis.set_label_position('top')

    for i, (idx, row) in enumerate(tax.iterrows()):
        sp, name, x, hue = row
        y = len(tax) - i - 1
        j.ax.text(x * 1.2, y, name, ha='left', va='center', zorder=100)
        # j.ax.plot([x, x], [y, 0], marker=None, color='gray', lw=.5, alpha=.4)

    j.ax.xaxis.grid(True, 'minor', linewidth=.4, alpha=.6, zorder=0)
    j.ax.xaxis.grid(True, 'major', linewidth=1, zorder=0)

    j.ax.fill_between([.9] + list(tax.n_ppis) + [1e6],
                      [28] + [len(tax) - k - 1 for k in range(len(tax))] + [-2],
                      len(tax) + 2,
                      zorder=10, color='w')
    j.tight_layout()
    j.figure.set_facecolor('white')
    # for sfx in ['png', 'pdf', 'svg']:
    #     j.savefig(f'interactome_sizes.{sfx}', dpi=300, transparent=False)
    return j


@mpl.style.context('seaborn-talk')
def plot_bias(plus: pd.DataFrame, minus: pd.DataFrame = None,
              bias: pd.DataFrame = None, ratio: float = 10.0,
              pal: bool = False) -> Figure:
    if minus is None:
        plus, minus = sep_plus_minus(plus)
    if bias is None:
        bias = estimate_bias_per_species(pd.concat((plus, minus)))

    s = 'species'
    bias = (bias.merge(plus.groupby(s)['label'].size(), on=s)
            .merge(minus.groupby(s)['label'].size(), on=s))
    bias.columns = ['species', 'bias', 'positives', 'negatives']
    bias['fulfill'] = bias.negatives / ratio / bias.positives
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    bias['bias'] = bias.bias.astype(float)

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = sns.scatterplot(data=bias,
                              ax=ax,
                              x='positives', y='negatives',
                              hue='bias', palette=cmap if pal else None,
                              # aspect=1, height=4,
                              s=40, alpha=.8,
                              zorder=99,
                              )

    # for i, point in bias.iterrows():
    #     if point.negatives < 10000:
    #         ax.annotate(point.species,
    #                     (point.positives * 1.4,
    #                      point.negatives * .9),
    #                     rotation=0, size=6)

    ax.set(xscale='log', yscale='log')
    ax.xaxis.grid(True, 'minor', linewidth=.2, alpha=.6, zorder=0)
    ax.yaxis.grid(True, 'minor', linewidth=.2, alpha=.6, zorder=0)
    ax.xaxis.grid(True, 'major', linewidth=.5, zorder=0)
    ax.yaxis.grid(True, 'major', linewidth=.5, zorder=0)
    sns.despine(fig, left=True, bottom=True)
    ax.set(box_aspect=1)
    ax.axline((1, ratio), (100, 100 * ratio), lw=.5,
              alpha=.6, zorder=1, color='black')
    ax.legend(frameon=False, bbox_to_anchor=(1, 1),
              title='bias', loc='upper left')
    fig.tight_layout()
    return fig


four = ['#D81B60', '#1E88E5', '#FFC107', '#004D40']
five_continous = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
eight = ['#000000', '#E69F00', '#56B4E9', '#009E73',
         '#F0E442', '#0072B2', '#D55E00', '#CC79A7']


def plot_degree_distribution(pairs: pd.DataFrame,
                             height: float = 2.4) -> Figure:
    pairs = fetch_degree_frequencies(pairs)
    cats = pairs.species.cat.categories
    pal = four if len(cats) <= 4 else eight

    fg = sns.relplot(data=pairs,
                     s=10, alpha=.8,
                     x='degree',
                     y='frequency',
                     hue='species',
                     # legend=False,
                     aspect=1,
                     height=height,
                     palette=dict(zip(cats, pal)) if len(cats) <= 8 else 'colorblind',
                     )
    fg.set(xscale='log', yscale='log', box_aspect=1)
    sns.despine(left=True, bottom=True)
    fg.tight_layout()
    return fg


def plot_ratio_degree(positives: pd.DataFrame,
                      negatives: pd.DataFrame = None,
                      ratio: float = 10,
                      taxonomy: pd.DataFrame = None,
                      rasterized: bool = False,
                      ) -> tuple[Figure, Union[None, Figure],
                                 pd.DataFrame, Union[None, pd.DataFrame]]:
    """
    Calculate the similarity between two sets of protein pairs:
    the Spearman or Pearson correlation coefficient between their
    protein-appearance frequency vectors.
    """
    # TODO test with homodimers to make sure this does not return
    #  node *degrees*, but the ratio of negative to positive edges
    if negatives is None:
        positives, negatives = sep_plus_minus(positives)
    plus, minus = [dict(zip(*np.unique(ar.iloc[:, [0, 1]], return_counts=True)))
                   for ar in (positives, negatives)]
    new_negatives = minus.keys() - plus.keys()
    minus.update({k: 0 for k in plus.keys() - minus.keys()})
    sp_lookup = positives[['hash_A', 'hash_B', 'species']].melt(
        id_vars='species')[['value', 'species']].drop_duplicates().set_index(
        'value').to_dict()['species']
    df = pd.DataFrame.from_records([(k, v, minus[k] / v, sp_lookup[k])
                                    for k, v in plus.items()],
                                   columns=['crc_hash', 'degree', 'ratio', 'species'])
    df['species'] = pd.Categorical(df.species)
    g = sns.JointGrid(data=df, x='ratio', y='degree', hue='species',
                      marginal_ticks=True, height=5, space=0, ratio=6,
                      palette='colorblind')
    g.plot_joint(sns.scatterplot, legend=False,
                 s=25, alpha=.8, rasterized=rasterized,
                 )
    g.plot_marginals(sns.kdeplot, warn_singular=False)
    g.ax_joint.set(  # xlim=(ratio - 4, ratio + 4),
        yscale='log', xscale='linear',
        xlabel='ratio −:+ interactions',
        ylabel='node degree')
    # g.ax_joint.axvline(x=ratio, lw=1, alpha=.5, zorder=0)
    # g.ax_marg_x.axvline(x=ratio, lw=1, alpha=.5, zorder=0)
    g.refline(x=ratio, lw=1, alpha=.5, zorder=0)
    g.ax_marg_x.set(yticks=[])
    g.ax_marg_y.set(xticks=[])
    g.figure.set_facecolor('white')
    # sns.despine(ax=g.ax_joint, left=False, bottom=False, right=False, top=False)
    sns.despine(ax=g.ax_marg_x, left=True)
    sns.despine(ax=g.ax_marg_y, bottom=False)

    if not new_negatives:
        return g.figure, None, df, None

    nsp = (negatives[['hash_A', 'hash_B', 'species']]
           .melt(id_vars='species', value_name='crc_hash')
           [['crc_hash', 'species']].value_counts().reset_index()
           .rename(columns={0: 'degree'}))
    nsp['species'] = pd.Categorical(nsp.species)
    nsp['kind'] = nsp.crc_hash.apply(lambda crc: 'proteome'
    if crc in new_negatives else 'interactome')

    psp = (positives[['hash_A', 'hash_B', 'species']]
           .melt(id_vars='species', value_name='crc_hash')
           [['crc_hash', 'species']].value_counts().reset_index()
           .rename(columns={0: 'degree'}))
    psp['species'] = pd.Categorical(psp.species)
    psp['kind'] = 'positives'
    nsp = pd.concat((nsp, psp))

    med_degrees = nsp.loc[nsp.kind == 'interactome'].groupby(
        'species')['degree'].median().to_dict()
    avg_degrees = nsp.loc[nsp.kind == 'interactome'].groupby(
        'species')['degree'].mean().to_dict()
    if taxonomy is None:
        order = sorted(set(nsp.species))
    else:
        order, names = [list(ar) for ar in taxonomy.loc[taxonomy.species.isin(
            set(nsp.species)), ['species', 'name']].values.T]

    h = sns.catplot(data=nsp,
                    x='degree',
                    y='species',
                    hue='kind',
                    dodge='True',
                    order=order,
                    orient='h',
                    jitter=.2,
                    height=5,
                    aspect=1,
                    s=1.4,
                    alpha=.3,
                    # palette='colorblind',
                    legend=False,
                    rasterized=True,
                    )
    h.set(xscale='log', box_aspect=1, xlabel='pairs per protein')
    for y, sp in enumerate(order):
        h.ax.text(avg_degrees[sp], y, '|', va='center', ha='center',
                  fontsize='large', fontweight='black')
        h.ax.text(med_degrees[sp], y, '·', va='center', ha='center',
                  fontsize='large', fontweight='black')

    h.ax.legend(frameon=False, title='', loc=(.15, 1), ncol=3, markerscale=.5)
    h.set(xlabel='pairs per protein')
    if taxonomy is not None:
        h.set(ylabel='', yticklabels=names)

    bo = False
    h.despine(left=bo, top=bo, right=bo, bottom=bo)
    return g.figure, h, df, nsp


def plot_plus_minus_degrees(plus: pd.DataFrame, minus: pd.DataFrame = None,
                            rasterized: bool = True,
                            ratio: float = 1.0,
                            ) -> Figure:
    if minus is None:
        plus, minus = sep_plus_minus(plus)

    deg = fetch_degrees(plus).merge(
        fetch_degrees(minus), on=['species', 'crc_hash'], how='left')

    deg['species'] = pd.Categorical(deg.species)
    g = sns.JointGrid(data=deg, x='degree_x', y='degree_y', hue='species',
                      marginal_ticks=True, height=5, space=0, ratio=6,
                      palette='colorblind')
    g.plot_joint(sns.scatterplot, legend=False,
                 s=25, alpha=.8, rasterized=rasterized,
                 )
    g.plot_marginals(sns.kdeplot, hue='species', warn_singular=False)
    g.ax_joint.set(  # xlim=(ratio - 4, ratio + 4),
        yscale='linear', xscale='log',
        xlabel='positives',
        ylabel='negatives')

    g.ax_marg_x.set(yticks=[])
    g.ax_marg_y.set(xticks=[])
    g.figure.set_facecolor('white')
    # sns.despine(ax=g.ax_joint, left=False, bottom=False, right=False, top=False)
    sns.despine(ax=g.ax_marg_x, left=True)
    sns.despine(ax=g.ax_marg_y, bottom=False)
    return g.figure


def plot_degrees_wide(ppis: pd.DataFrame,
                      negs: pd.DataFrame,
                      tax: pd.DataFrame = None,
                      dodge: bool = False,
                      species: list = None,
                      height: float = 2.5,
                      aspect: float = 4.0) -> Figure:
    tdf = (fetch_degrees(negs).merge(fetch_degrees(ppis),
                                     on=['crc_hash', 'species'], how='left')
           .rename(columns=dict(degree_x='negatives', degree_y='positives')))
    tdf['kind'] = tdf.positives.apply(lambda p: 'proteome' if pd
                                      .isna(p) else 'interactome')
    tdf.positives = tdf.positives.fillna(0)
    tdf = tdf.convert_dtypes()
    tdf['degree'] = tdf.negatives + tdf.positives

    if species:
        tdf = tdf.loc[tdf.species.isin(species)]
        tdf.species = tdf.species.cat.remove_unused_categories()

    if tax is None:
        order = list(tdf.species.value_counts().index)
    else:
        order, names = [list(ar)[::-1] for ar in tax.loc[tax.species.isin(
            set(tdf.species)), ['species', 'name']].values.T]

    h = sns.catplot(data=tdf,
                    x='degree',
                    y='species',
                    hue='kind',
                    dodge=dodge,
                    order=order,
                    orient='h',
                    jitter=.3,
                    height=height,
                    aspect=aspect,
                    s=2.4,
                    alpha=.3,
                    # palette='colorblind',
                    legend=False,
                    rasterized=True,
                    )
    h.set(xscale='log', box_aspect=1 / aspect, xlabel='pairs per protein')
    h.ax.legend(frameon=False, title='', markerscale=.5, loc=(.15, 1), ncol=2)
    if tax is not None:
        h.set(ylabel='', yticklabels=names)

    bo = True
    h.despine(left=bo, top=bo, right=bo, bottom=bo)
    h.tight_layout()
    return h


def plot_ratio_grids(df: pd.DataFrame, order: list = None, ratio: float = 10.0) -> tuple[Figure, Figure]:
    g = sns.relplot(data=df,
                    x='ratio', y='degree',  # hue='species',
                    col='species', col_wrap=5,
                    col_order=order,
                    height=1.5,
                    alpha=.4,
                    )
    g.set(box_aspect=1, yscale='log', xlabel='', ylabel='')
    g.refline(x=ratio, lw=1, alpha=.5, zorder=0)
    g.set_titles(col_template='{col_name}')
    g.tight_layout()
    g.figure.set_facecolor('white')
    g.axes[-5].set(xlabel='ratio -:+ interactions', ylabel='degree')
    # g.savefig('train_ratio_degree_grid.png', dpi=300, transparent=False)

    h = sns.displot(data=df, kind='hist',
                    x='ratio', y='degree',  # hue='species',
                    col='species', col_wrap=5,
                    col_order=order,
                    height=1.5,
                    # alpha=.4,
                    # rug=True,
                    stat='density',
                    pmax=.07,
                    bins=40,
                    discrete=(True, False),
                    log_scale=(False, True)
                    )
    h.set(box_aspect=1,  # yscale='log',
          xlabel='', ylabel='')
    # h.refline(x=c.ratio, lw=1, alpha=.5, zorder=0)
    h.set_titles(col_template='{col_name}')
    h.tight_layout()
    h.figure.set_facecolor('white')
    h.axes[-5].set(xlabel='ratio -:+ interactions', ylabel='degree')
    # h.savefig('train_ratio_degree_grid_hist.png', dpi=300, transparent=False)

    return g, h


@mpl.style.context('seaborn')
def plot_test_ratios(test_all: pd.DataFrame, ratio: float = 10.0,
                     flip: bool = False) -> Figure:
    pairs = fetch_ratios(test_all)
    species = list(pairs.species.value_counts().index)[:5]

    shape = (len(species), 3)
    size = (8, 3 * len(species))
    if flip:
        shape, size = shape[::-1], size[::-1]

    fig, axes = plt.subplots(*shape, figsize=size,
                             sharey='row' if not flip else 'col',
                             sharex='row' if not flip else 'col')
    # if shape[0] == 1:
    #     axes = axes.reshape(axes.shape[0], 1)
    for row, sp in enumerate(species):
        # ymax = max(pairs.loc[(pairs.species == sp)
        #                      & (pairs.label == 1) & (pairs.x == 0), 'degree'])
        # xmax = max(pairs.x)
        # return pairs.loc[pairs.species == sp]
        axes_now = (axes[row, :] if not flip else axes[:, row]) \
            if len(species) > 1 else axes
        for cclass, ax in enumerate(axes_now):
            dat = pairs.loc[(pairs.cclass == cclass + 1)
                            & (pairs.species == sp)]
            for label, (color, name) in enumerate(zip(
                    ('#D81B60', '#1E88E5'), ('−', '+'))):
                dat_ = dat.loc[dat.label == label]
                ax.plot(dat_.x, dat_.degree, color=color, label=name)
            ax.plot(dat_.x, ratio * dat_.degree, color='.5',
                    alpha=.5, zorder=0, label=f'{ratio} × +')
            ax.set(yscale='linear', xscale='log', box_aspect=1)
            #        xlim=(1, xmax),
            #        ylim=(1, int(ratio * ymax + 1)))
            if row == 0 and not flip:
                ax.set(title=f'C{cclass + 1}')
            elif row == len(species) - 1 and flip:
                ax.set_title(f'C{cclass + 1}', y=.5, loc='right')

    ax1, ax2 = (axes[-1, 0], axes[0, -1]) if len(species) > 1 else (axes[0], axes[-1])
    ax1.set(xlabel='degree rank', ylabel='node degree')
    ax2.legend(frameon=False)
    fig.subplots_adjust(hspace=.2 if flip else .05, wspace=.2 if flip else .1)
    # fig.tight_layout()
    raise DeprecationWarning('Proteins by themselves do not have a C123 '
                             'class but are assigned one here! '
                             'PPIs belong to a class.')
    return fig


@mpl.style.context('seaborn-whitegrid')
def plot_c_classes(df: pd.DataFrame) -> tuple[Figure, dict[int, int]]:
    fig, ax = plt.subplots(figsize=(4, 3), facecolor='white')
    bg = sns.countplot(
        x=df.cclass,
        palette={1: '#FF669D', 2: '#71BDFF', 3: '#FFE083'}, ax=ax)
    fg = sns.countplot(
        x=df.loc[df.label == 1, 'cclass'],
        palette={1: '#D81B60', 2: '#1E88E5', 3: '#FFC107'}, ax=ax)
    vc = df.loc[df.label == 1, 'cclass'].value_counts().sort_index()
    ax.bar_label(container=ax.containers[1], labels=vc)
    ax.set(box_aspect=1)
    sns.despine(ax=ax, left=True, bottom=True, right=True)
    a2b = lambda y: y / len(df.cclass)
    b2a = lambda y: len(df.cclass) * y
    ax2 = ax.secondary_yaxis('right', functions=(a2b, b2a))
    ax2.set(ylabel='share')
    sns.despine(ax=ax2, left=True, right=True)
    return fig, dict(vc)


# @mpl.style.context('seaborn-whitegrid')
def plot_interspecies_loss(ppis_without: pd.DataFrame,
                           ppis_with: pd.DataFrame,
                           taxonomy: pd.DataFrame = None
                           ) -> tuple[Figure, pd.DataFrame]:
    """
    Feed in two dataframes; one with and one without interspecies interactions.
    :param ppis_without:
    :param ppis_with:
    :param taxonomy:
    :return:
    """
    dk, dq = [d.species.value_counts().reset_index().rename(
        columns=dict(index='species', species='n_ppis')) for d in [ppis_without, ppis_with]]
    dq = dk.merge(dq, on='species', how='outer')
    dq = dq.fillna(0)

    dq['lost'] = dq.n_ppis_y - dq.n_ppis_x
    dq['share'] = dq.lost / dq.n_ppis_y
    dq = dq.merge(taxonomy[['species', 'name']], on='species', how='left')
    dq = dq.fillna('')
    dq.species = pd.Categorical(dq.species)
    dq = dq.convert_dtypes()

    # fig, axes = plt.subplots(1, 2, figsize=(4, 6), sharey=True, facecolor='None')
    # sns.pointplot(data=dq, x='lost', y='species',
    #               color='#1E88E5', scale=.5,
    #               ax=axes[0], join=False, legend=False)
    # axes[0].set(xscale='log', xlabel='absolute loss',
    #             ylabel=('' if taxonomy is not None else 'species'))
    #
    # sns.pointplot(data=dq, x='share', y='species',
    #               color='#D81B60', scale=.5,
    #               ax=axes[1], join=False, legend=False)
    # axes[1].set(xlabel='relative loss', ylabel='',
    #             xticks=[0, .25, .5, .75, 1],
    #             xticklabels=['0', '.25', '.5', '.75', '1'])
    # if taxonomy is not None:
    #     axes[1].set(yticklabels=list(taxonomy.loc[taxonomy.species.isin(
    #         dq.species)].sort_values(by='species')['name']))
    # sns.despine(fig, left=True, bottom=True)
    # axes[0].grid(zorder=0)
    # axes[1].grid(zorder=0)

    fig, ax = plt.subplots(figsize=(8, 3), facecolor='None')
    scatter = sns.scatterplot(data=dq,
                              x='lost',
                              y='share',
                              s=40,
                              ax=ax,
                              )
    ax.set(xscale='log',  # ylim=(None, 1),
           xlabel='number of inter-species PPIs per species dataset',
           ylabel='inter-species share',
           )
    for i, point in enumerate(dq.itertuples()):
        ax.annotate(point.species,
                    (point.lost * 1.05,
                     point.share + .03),
                    rotation=50, size=6, zorder=0)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig, dq
