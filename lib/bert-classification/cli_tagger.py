import argparse
import logging
import os
import sys
import json
import glob
import jsonlines
from dask.diagnostics import ProgressBar
import dask.bag as db
from smart_open import smart_open
from typing import Optional

from dask.distributed import Client


# from inference import predict_entities

from utils import Timer, chunk

logger = logging.getLogger(__name__)


def predict_mentions_test(content_items):
    result_json = []
    for ci in content_items:
        entity_json = {
            "entity": "newsag",
            "name": "Reuters",
            "lOffset": 2637,
            "rOffset": 2642,
            "id": ci["id"] + ":2637:2642:newsag:bert"
        }
        result_json.append(entity_json)
    return result_json


def run_newsagency_tagger(input_dir: str,
                          output_dir: str,
                          prefix: Optional[str] = None) -> None:
    t = Timer()
    if prefix is not None:
        path = f"{input_dir}/{prefix}*.jsonl.bz2"
    else:
        path = f"{input_dir}/*.jsonl.bz2"

    logger.info(f"Indexing files in {path}")

    files = glob.glob(path)
    logger.info(f'Number of files: {len(files)}.')

    batches = list(chunk(files, 10))
    total = len(batches)
    for i, b in enumerate(batches):
        logger.info(f'Parsing {i}/{total}: {b}')
        bag_articles = db.read_text(b) \
            .filter(lambda s: len(s) > 2) \
            .map(json.loads) \
            .filter(lambda ci: ci['tp'] == 'ar')

        bag_mentions = bag_articles.map_partitions(predict_mentions_test) \
            .map(json.dumps)

        with ProgressBar():
            # print(bag_articles.take(2))
            bag_mentions.to_textfiles(f'{output_dir}/' + '*.jsonl.bz2',
                                      name_function=lambda x: str(x))
        logger.info(f"Batch time: {t.tick()}")


def parse_args():
    description = ""
    epilog = ""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument(
        "--input_dir",
        dest="input_dir",
    )

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
    )

    parser.add_argument(
        "-l",
        "--logfile",
        dest="logfile",
        help="write log to FILE",
        metavar="FILE"
    )

    parser.add_argument(
        "--verbose",
        dest="verbose",
        default=3,
        type=int,
        metavar="LEVEL",
        help="set verbosity level: 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO 4=DEBUG (default %(default)s)",
    )

    parser.add_argument(
        "--prefix",
        dest="prefix",
        default="000",
        type=str,
        help="todo",
    )

    return parser.parse_args()


if __name__ == "__main__":

    arguments = parse_args()

    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]

    handlers = [logging.StreamHandler()]
    if arguments.logfile:
        handlers.append(logging.FileHandler(arguments.logfile, mode="w"))

    logging.basicConfig(
        level=log_levels[arguments.verbose],
        format="%(asctime)-15s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    client = Client('127.0.0.1:8000')

    run_newsagency_tagger(arguments.input_dir, arguments.output_dir, arguments.prefix)
