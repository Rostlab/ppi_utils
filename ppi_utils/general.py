import hashlib
import json
import shutil
import subprocess as cmd
import zipfile
from Bio.Seq import Seq
from Bio.SeqUtils.CheckSum import crc64
from pathlib import Path
from typing import Iterable, Union, IO


def cmd_run(args, verbose=False):
    stdout = cmd.DEVNULL
    if verbose:
        stdout = None
        print(f'\033[94m' + ' '.join(args) + '\033[0m')
    cmd.run(args, stdout=stdout, stderr=cmd.STDOUT)


def values_sorted_by_key(d: dict):
    return (d[k] for k in sorted(d.keys()))


def file_hash(file_path):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)

    with open(file_path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


class DoesNotContain(Exception):
    pass


def glob_type(directory: Union[str, Path],
              suffix: str, relax: bool = False,
              recursive: bool = False) -> set[Path]:
    """
    Glob inside a directory for files matching the given suffix.
    Throws an exception if no matches are found and not told to relax.
    Tailing whitespace is removed from the passed file suffix,
    but the left side is not modified.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError
    suffix = suffix.rstrip()
    func = directory.glob if not recursive else directory.rglob
    files = set(func(f'*{suffix}'))
    if not files:
        if relax:
            return set()
        raise DoesNotContain(f'No {suffix} files found in {directory}')
    return files


def glob_types(directory: Union[str, Path],
               suffixes: Iterable,
               recursive: bool = False) -> set[Path]:
    files = set()
    for suffix in suffixes:
        try:
            files |= glob_type(directory, suffix, recursive=recursive)
        except DoesNotContain:
            pass
    if not files:
        raise DoesNotContain(
            f'No {"|".join(suffixes)} files found in {directory}')
    return files


def run_uniqueprot(input_file: Union[str, Path],
                   output_file: Union[str, Path],
                   hval_config: Path,
                   pretend: bool = False,
                   verbose: bool = True):
    args = ['rostclust', 'uniqueprot',
            '--hval-config-path', str(hval_config),
            str(input_file), str(output_file)]
    if pretend:
        return ' '.join(args)
    elif Path(input_file).stat().st_size == 0:
        print(f'{input_file} is empty, creating dummy!')
        shutil.copy(input_file, output_file)
    else:
        cmd_run(args, verbose=verbose)


def run_uniqueprot2D(input_file: Union[str, Path],
                     database_file: Union[str, Path],
                     output_file: Union[str, Path],
                     hval_config: Path,
                     pretend: bool = False,
                     verbose: bool = True):
    args = ['rostclust', 'uniqueprot2d',
            '--hval-config-path', str(hval_config),
            str(input_file), str(database_file), str(output_file)]
    if pretend:
        return ' '.join(args)
    else:
        cmd_run(args, verbose=verbose)


def get_seq_hash(seq: Union[str, Seq]) -> str:
    return crc64(seq)


def to_fasta(_id: Union[str, int], seq: str, file_handle: IO) -> None:
    _ = file_handle.write(''.join(to_lines(_id, seq)))


def write_json(_dict: dict, _json_path: Union[str, Path]) -> dict:
    with Path(_json_path).open('w') as json_file:
        json.dump(_dict, json_file, indent=2)
    return _dict


def read_json(_json_path: Union[str, Path]) -> dict:
    return {int(k): v for k, v in json.load(
        Path(_json_path).open('r')).items()}


def to_lines(_id: str, seq: Union[str, Seq], lw: int = 60) -> str:
    yield f'>{_id}\n'
    i = 0
    seq = str(seq)
    while i < len(seq):
        yield seq[i:i + lw] + '\n'
        i += lw


def clean(data_dir: Union[str, Path], name: Union[str, Path] = None) -> None:
    data_dir = Path(data_dir)
    assert data_dir.is_dir()

    if not name:
        name = data_dir.stem

    (data_dir / 'slurm_logs').mkdir(exist_ok=True)
    for log in data_dir.glob('slurm-*.out'):
        log.rename(data_dir / 'slurm_logs' / log.name)

    with zipfile.ZipFile(data_dir.parent / f'{name}.zip',
                         'w', zipfile.ZIP_DEFLATED) as zf:
        for sfx in ['tsv', 'fasta']:
            zf.write(data_dir / f'apid_train.{sfx}', f'apid_train.{sfx}')
            zf.write(data_dir / f'apid_validation.{sfx}', f'apid_validation.{sfx}')
            zf.write(data_dir / f'huri_test.{sfx}', f'huri_test.{sfx}')
        if (f := Path(data_dir / 'crc_hashes.tsv')).is_file():
            zf.write(f, f.name)
        for d in ['val', 'test']:
            if (f := data_dir / f'{d}_cclasses.svg').is_file:
                zf.write(f, f'plots/{f.name}')
        for d in ['train', 'val', 'test']:
            if (f := data_dir / f'{d}_ratio_degree.svg').is_file():
                zf.write(f, f'plots/{f.name}')
            if (f := data_dir / f'{d}_bias.svg').is_file():
                zf.write(f, f'plots/{f.name}')

        nb = list(data_dir.parents[1].rglob(f'{data_dir.name}.ipynb'))
        if len(nb) != 1:
            print('found no clearly matching notebook')
        else:
            zf.write(nb[0], f'{name}.ipynb')

        # for d in ['train', 'val', 'test']:
        #     if (f := data_dir / f'{d}_proteome.json').is_file():
        #         zf.write(f, f'proteomes/{f.name}')
    print(f'done: {zf.filename}')
