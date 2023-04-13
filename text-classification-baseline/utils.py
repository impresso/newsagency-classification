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

def get_custom_logger(name: str,
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
        response = urllib.request.urlopen(url) # TODO: dangerous
        return response.read().decode('utf-8')

    elif path:
        with open(path) as f:
            return f.read()

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

    tsv_lines = [l.split('\t') for l in get_tsv_data(tsv_path, tsv_url).split('\n')]
    label_col_number = tsv_lines[0].index(labels_column)
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j]:
                assert tsv_lines[tsv_line_numbers[i][j]][0] == words[i][j]
                tsv_lines[tsv_line_numbers[i][j]][label_col_number] = labels[i][j]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['\t'.join(l) for l in tsv_lines]))


