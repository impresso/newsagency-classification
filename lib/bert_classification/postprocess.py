#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Small piece to add Wikidata QID to detected news agency mentions and replace sentence offsets with article offsets in ci ID.
Exec: python postprocess.py --input_dir=/scratch/newsagency-project/na_mentions --output_dir=/scratch/newsagency-project/na_mentions_postprocessed --logfile=log-test.log [--prefix=XX]
(no prefix to parse all)
"""

import json
import argparse
import logging
import glob
from dask.diagnostics import ProgressBar
import dask.bag as db
from typing import Optional
from dask.distributed import Client
from typing import Dict, List

logger = logging.getLogger(__name__)

WIKIDATA_IDs = {
    "AFP": "Q40464",
    "ANP": "Q966898",
    "ANSA": "Q392934",
    "AP": "Q40469",
    "APA": "Q680662",
    "ATS-SDA": "Q430109",
    "Belga": "Q815453",
    "BTA": "Q2031809",
    "CTK": "Q341118",
    "DDP-DAPD": "Q265330",
    "DNB": "Q1205856",
    "Domei": "Q2913752",
    "DPA": "Q312653",
    "Europapress": "NIL",
    "Extel": "Q1525848",
    "Havas": "Q2826560",
    "Interfax": "Q379271",
    "Kipa": "Q1522416",
    "PAP": "Q1484980",
    "Reuters": "Q130879",
    "SPK-SMP": "Q2256560",
    "Stefani": "Q1415690",
    "TANJUG": "Q371267",
    "TASS": "Q223799",
    "Telunion": "Q3517301",
    "TT": "Q1312158",
    "UP-UPI": "Q493845",
    "Wolff": "Q552226",
    "Xinhua": "Q204839",
}


def post_process(ci: List) -> List:
    for content_item in ci:
        label = content_item["entity"].split(".")[-1]
        if label in WIKIDATA_IDs:
            content_item["qid"] = WIKIDATA_IDs[label]
        if "lOffset" in content_item:
            del content_item["lOffset"]
        if "rOffset" in content_item:
            del content_item["rOffset"]
        ci_id_parts = content_item["id"].split(":")
        content_item[
            "id"] = f'{ci_id_parts[0]}:{content_item["lArticleOffset"]}:{content_item["rArticleOffset"]}:{ci_id_parts[3]}:{ci_id_parts[4]}'
    return ci


def run_newsagency_postprocess(input_dir: str,
                               output_dir: str,
                               man_prefix: Optional[str] = None) -> None:
    if not man_prefix:
        prefixes = [str(i).zfill(2) for i in range(0, 23)]
    else:
        prefixes = [man_prefix]
    total = len(prefixes)
    for i, prefix in enumerate(prefixes):
        path = f'{input_dir}/{prefix}/*.jsonl.bz2'
        logger.info(f'Parsing {i}/{total}')
        try:
            files = glob.glob(path)
            logger.info(f'Processing {files}')
            bag_articles = db.read_text(files) \
                .map(json.loads) \
                .map(post_process) \
                .map(json.dumps)
            with ProgressBar():
                bag_articles.to_textfiles(f'{output_dir}/{prefix}/*.jsonl.bz2')
        except ValueError as e:
            logger.info(f'Could not parse {path}; {e}')


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

    # client = Client('127.0.0.1:8000')

    run_newsagency_postprocess(
        arguments.input_dir,
        arguments.output_dir,
        arguments.prefix)

    # client.close()
