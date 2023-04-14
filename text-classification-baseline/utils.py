"""
@ hipe-eval / HIPE-2022-baseline
A few general utilities for transformers_baseline.
"""

import random
import logging
from typing import List, Union

import numpy as np
import urllib.request
from typing import Set, List, Union, NamedTuple, Dict, Optional
import torch

from dataset import COLUMNS

SEED = 42


def set_seed(seed) -> None:
    # np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def get_custom_logger(
        name: str,
        level: int = logging.INFO,
        fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt: str = '%Y-%m-%d %H:%M'):
    """Custom logging wraper, called each time a logger is declared in the package."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    """Sets seed for `random`, `np.random` and `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


logger = get_custom_logger(__name__)


def get_tsv_data(path: Optional[str] = None, url: Optional[str] = None) -> str:
    """Fetches tsv data from a path or an url."""

    assert path or url, """`path` or `url` must be provided"""

    if url:
        response = urllib.request.urlopen(url)  # TODO: dangerous
        return response.read().decode('utf-8')

    elif path:
        with open(path) as f:
            return f.read()


def write_predictions(tsv_dataset, words_list, preds_list):
    """
    @param tsv_dataset:
    @param words_list:
    @param preds_list:
    @return:
    """
    with open(tsv_dataset, 'r') as f:
        tsv_lines = f.readlines()

    flat_words_list = [item for sublist in words_list for item in sublist]
    flat_preds_list = [item for sublist in preds_list for item in sublist]
    with open(tsv_dataset.replace('.tsv', '_pred.tsv'), 'w') as f:
        idx = 0
        for idx_tsv_line, tsv_line in enumerate(tsv_lines):
            if idx_tsv_line == 0:
                f.write(tsv_line)
            elif len(tsv_line.split('\t')) != len(COLUMNS):
                f.write(tsv_line)
            elif len(tsv_line.strip()) == 0:
                f.write(tsv_line)
            else:
                try:
                    f.write(
                        flat_words_list[idx] +
                        '\t' +
                        flat_preds_list[idx] +
                        '\n')
                except BaseException:
                    import pdb
                    pdb.set_trace()
                idx += 1
                f.flush()


def write_predictions_to_tsv(words: List[List[Union[str, None]]],
                             labels: List[List[Union[str, None]]],
                             tsv_line_numbers: List[List[Union[int, None]]],
                             output_file: str,
                             labels_column: str,
                             tsv_path: str = None,
                             tsv_url: str = None, ):
    """Get the source tsv, replaces its labels with predicted labels and write a new file to `output`.

    `words`, `labels` and `tsv_line_numbers` should be three aligned list, so as in HipeDataset.
    """

    logger.info(f'Writing predictions to {output_file}')

    tsv_lines = [l.split('\t')
                 for l in get_tsv_data(tsv_path, tsv_url).split('\n')]
    label_col_number = tsv_lines[0].index(labels_column)
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j]:
                assert tsv_lines[tsv_line_numbers[i][j]][0] == words[i][j]
                tsv_lines[tsv_line_numbers[i][j]
                          ][label_col_number] = labels[i][j]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['\t'.join(l) for l in tsv_lines]))
