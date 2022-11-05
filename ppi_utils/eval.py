import warnings
from itertools import product
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.utils import resample
from tqdm import tqdm

from ppi_utils.general import glob_type

mpl.use("Agg")


def parse_predictions_from_folder(
        path: Union[str, Path],
        pattern: str = '.predictions.tsv') -> pd.DataFrame:
    """
    Glob inside a given folder for '.predictions.tsv'
    files and return them as a dataframe
    :param path:
    :param pattern:
    :return:
    """
    tsv_files = glob_type(path, pattern, recursive=True)

    def read_tsv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t', header=None, names=['n0', 'n1', 'label', 'pred'])
        df['species'], df['epoch'], *_ = path.name.split('.')[0].split('_') + ['10']
        df.epoch = df.epoch.astype(int)
        df.label = df.label.astype(int)
        return df

    df = pd.concat([read_tsv(f) for f in tsv_files])
    tsv_path = Path(path) / 'all_predictions.tsv'
    print(f'Writing non-bootstrapped predictions table to {tsv_path}')
    df.to_csv(tsv_path, sep='\t', header=False)
    return df


def parse_predictions_alternative(
        path: Union[str, Path],
        pattern: str = '_predictions.tsv') -> pd.DataFrame:
    """
    Glob inside a given folder for '_predictions.tsv'
    files and return them as a dataframe
    :param path:
    :param pattern:
    :return:
    """
    tsv_files = glob_type(path, pattern, recursive=True)

    def read_tsv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['n0', 'n1', 'label', 'p_bin', 'p0', 'p1'])
        df['species'], df['epoch'], *_ = path.name.split('.')[0].split('_') + ['10']
        df.epoch = df.epoch.astype(int)
        df.label = df.label.astype(int)
        return df

    df = pd.concat([read_tsv(f) for f in tsv_files])
    tsv_path = Path(path) / 'all_preds.tsv'
    print(f'Writing non-bootstrapped predictions table to {tsv_path}')
    df.to_csv(tsv_path, sep='\t', header=False)
    return df


def parse_predictions_alternative_two(
        path: Union[str, Path],
        pattern: str = '_predictions.tsv') -> pd.DataFrame:
    """
    Glob inside a given folder for '_predictions.tsv'
    files and return them as a dataframe
    :param path:
    :param pattern:
    :return:
    """
    tsv_files = glob_type(path, pattern, recursive=True)

    def read_tsv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['n0', 'n1', 'label', 'p_bin', 'p0', 'p1'])
        df['species'] = path.name.split('_')[0]
        ix = path.name.find('epoch')
        df['epoch'] = path.name[ix + 5:ix + 6]
        df.epoch = df.epoch.astype(int)
        df.label = df.label.astype(int)
        return df

    df = pd.concat([read_tsv(f) for f in tsv_files])
    tsv_path = Path(path) / 'all_preds.tsv'
    print(f'Writing non-bootstrapped predictions table to {tsv_path}')
    df.to_csv(tsv_path, sep='\t', header=False)
    return df


def bootstrap_dataframe(df: pd.DataFrame, bootstraps: int = 1) -> [pd.DataFrame]:
    """
    # possibly slice from a notebook before

    :param df:
    :param bootstraps:
    :return:
    """
    assert df.columns.tolist() == ['n0', 'n1', 'label', 'pred', 'species', 'epoch']
    all_data = dict()

    for species, epoch in product(df.species.unique(), df.epoch.unique()):
        slice = df.loc[(df.species == species) & (df.epoch == epoch), ['label', 'pred']]
        if slice.empty:
            continue
        # print(species, epoch)
        errors = dict()

        labels, predictions = slice.T.values
        # precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
        # instead, precision and recall as columns:
        with warnings.catch_warnings(record=True) as caught:
            prs = [np.vstack(precision_recall_curve(labels, predictions)[:2]).T]
            auprs = [average_precision_score(labels, predictions)]
            diffs = [auc(*[*precision_recall_curve(labels, predictions)[:2]][::-1]) - auprs[-1]]
            rocs = [np.vstack(roc_curve(labels, predictions)[:2]).T]
            try:
                aurocs = [roc_auc_score(labels, predictions)]
            except ValueError as ve:
                warnings.warn(str(ve))

        if caught:
            bk = '\n - '
            warnings.warn(f'{species}, epoch {epoch}: skip bootstrapping '
                          f'due to the following error(s): '
                          f'\n - {f"{bk}".join({str(w.message) for w in caught})}',
                          RuntimeWarning)
            continue

        for j in tqdm(range(1, bootstraps), desc=f'{species}_{epoch}'):
            labels, predictions = resample(slice, random_state=j).T.values
            # prs.append(np.vstack(precision_recall_curve(labels, predictions)[:2]).T)
            aupr = average_precision_score(labels, predictions)
            auprs.append(aupr)
            diff = auc(*[*precision_recall_curve(labels, predictions)[:2]][::-1]) - aupr
            diffs.append(diff)
            # rocs.append(np.vstack(roc_curve(labels, predictions)[:2]).T)
            try:
                aurocs.append(roc_auc_score(labels, predictions))
            except ValueError as ve:
                aurocs.append(0)
                desc = ve.args[0]
                errors[desc] = errors.get(desc, 0) + 1

        if errors:
            warnings.warn(f'{species} epoch {epoch} had errors: {errors}', RuntimeWarning)

        all_data[f'{species}_{epoch}'] = [np.vstack(a) for a in [prs, auprs, rocs, aurocs, diffs]]

    prs, aucs, rocs, diffs = list(), list(), list(), list()
    for k, v in all_data.items():
        species, epoch = k.split('_')
        precision, recall = v[0].T
        fpr, tpr = v[2].T
        prs.append(pd.DataFrame({'species': species, 'epoch': int(epoch),
                                 'precision': precision, 'recall': recall}))
        rocs.append(pd.DataFrame({'species': species, 'epoch': int(epoch),
                                  'fpr': fpr, 'tpr': tpr}))
        aucs.append(pd.DataFrame({'species': species, 'epoch': int(epoch),
                                  'aupr': v[1].flatten(), 'auroc': v[3].flatten()}))
        diffs.append(pd.DataFrame({'species': species, 'epoch': int(epoch),
                                   'diffs': v[-1].flatten()}))
    prs, aucs, rocs, diffs = [pd.concat(a) for a in [prs, aucs, rocs, diffs]]

    # calculate mean and standard error -> 1.96 * std because seaborn is messed up
    (1.96 * aucs.groupby(['species', 'epoch']).std()).rename(
        dict(aupr='aupr_se', auroc='auroc_se'), axis='columns')
    aucs = (aucs.groupby(['species', 'epoch']).mean().rename(
        dict(aupr='aupr_mean', auroc='auroc_mean'), axis='columns')
    ).join(other=
    (1.96 * aucs.groupby(['species', 'epoch']).std()).rename(
        dict(aupr='aupr_se', auroc='auroc_se'), axis='columns'),
        on=['species', 'epoch']
    ).reset_index()

    return prs, aucs, rocs, diffs


def plot_eval_predictions(labels, predictions, path="figure"):
    """
    Plot histogram of positive and negative predictions, precision-recall curve, and receiver operating characteristic curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param path: File prefix for plots to be saved to [default: figure]
    :type path: str
    """

    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    # ax1.set_xlim(0, 1)  # maybe this will make ...
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    # ax2.set_xlim(0, 1)  # ... the histograms actually interpretable
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(path + ".phat_dist.png")
    plt.close()

    precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    print("AUPR:", aupr)

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall (AUPR: {:.3})".format(aupr))
    plt.savefig(path + ".aupr.png")
    plt.close()

    fpr, tpr, roc_thresh = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    print("AUROC:", auroc)

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Receiver Operating Characteristic (AUROC: {:.3})".format(auroc))
    plt.savefig(path + ".auroc.png")
    plt.close()
