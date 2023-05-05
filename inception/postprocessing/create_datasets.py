"""
...

Usage:
    create_datasets.py --log-file=<log> --input-dir=<id> --output-dir=<od> --data-version=<v> --annotation-planning=<ann> [--discarded-dir=<dis>] [--language=<lang>]
"""  # noqa

from docopt import docopt
from pathlib import Path
import logging
import os
import pandas as pd
import random
from typing import List, Optional
from helpers import is_tsv_complete

LOGGER = logging.getLogger(__name__)



def concat_tsv_files(output_path: str, input_files: List[str]) -> None:
    """ 
    Takes list of tsv input files and concatenates them
    """
    with open(output_path, "w", encoding="utf-8", newline="") as out_tsv_file:
        data = []
        for n, file in enumerate(input_files):

            if not os.path.exists(file):
                LOGGER.warning(f"Input file {file} does not exist")
                continue

            LOGGER.debug(f"Read input from file {file}")

            with open(file, "r", encoding="utf-8") as inp_tsv_file:
                if n > 0:
                    lines = inp_tsv_file.readlines()[1:]
                else:
                    lines = inp_tsv_file.readlines()[0:]
                data.append("".join(lines))

        out_tsv_file.write("\n".join(data))


def create_sample(input_dir, output_dir, version):

    langs = ["fr", "de"]

    for lang in langs:
        corpus_input_dir = os.path.join(input_dir, lang, "tsv")
        output_path = os.path.join(
            output_dir, version, lang, f"newsagency-data-{version}-sample-{lang}.tsv"
        )
        logging.info(output_path)

        files = [
            os.path.join(corpus_input_dir, file)
            for file in os.listdir(corpus_input_dir)
        ]
        logging.info(f"Found {len(files)} .tsv files in {corpus_input_dir}")

        with open(output_path, "w", encoding="utf-8", newline="") as out_tsv_file:
            data = []
            for n, file in enumerate(files):
                with open(file, "r", encoding="utf-8") as inp_tsv_file:
                    if n > 0:
                        lines = inp_tsv_file.readlines()[1:]
                    else:
                        lines = inp_tsv_file.readlines()[0:]
                    data.append("".join(lines))

            out_tsv_file.write("\n".join(data))


def create_datasets(input_dir: str, output_dir: str, ann_planning: str, version: str, 
                    discarded_dir: str, language: Optional[str]) -> None:
    """ 
    Create a tsv dataset file for each (or selected) language (de, fr) and split (train, dev, test)
    
    :param input_dir: base directory where input tsv files are stored (should include folder "annotated_retok_autosegment")
    :param output_dir: directory where concatenated tsv files are stored
    :param ann_planning: path to the document containing the annotation planning (one row per doc)
    :param version: version of data, will be included in filename
    """
    def derive_document_path(row, input_dir, lang):
        if row["Document ID"]:
            if row["Document ID"] is None:
                return None
            if ".xmi" in row["Document ID"]:
                document_name = row["Document ID"].replace(".xmi", ".tsv")
            elif ".txt" in row["Document ID"]:
                document_name = row["Document ID"].replace(".txt", ".tsv")
            else:
                document_name = row["Document ID"] + ".tsv"

            annotation_dir = "annotated_retok_autosegment"

            return os.path.join(input_dir, annotation_dir, lang, "tsv", document_name)
        else:
            return None

    if language:
        langs = [language]
    else:
        langs = ["de", "fr"]

    splits = ["train", "dev", "test"]
    basedir = os.path.join(output_dir, version)

    if not os.path.exists(basedir):
        LOGGER.info(f"Created folder {basedir} as it did not exist")
        Path(basedir).mkdir(parents=True, exist_ok=True)

    for lang in langs:

        #check if there are discarded files (due to too bad OCR)
        discarded =[]
        if discarded_dir:
            discarded_file = discarded_dir + "/" + f"discarded_{lang}.txt"
            #check if file exists for this language
            if Path(discarded_file).is_file():
                with open(Path(discarded_file), "r") as f:
                    discarded = [line.strip() for line in f.readlines()]

        assignments_df = pd.read_csv(
                ann_planning
        )
        assignments_df = assignments_df[
            (assignments_df["Corpus"] == lang)
            & (assignments_df["Document ID"].notnull())
            & (assignments_df["Finished Annotation"] == True)
            & (~assignments_df["Document ID"].isin(discarded))
        ]

        # add path of each document in the dataframe
        # if it's marked as isMiniRef, then it's different
        assignments_df["Path"] = assignments_df.apply(
            lambda row: derive_document_path(row, input_dir, lang), axis=1
        )

        for split in splits:

            document_paths = list(assignments_df[assignments_df.split == split].Path)
            dataset_path = create_dataset(
                document_paths, lang, split, version, output_dir
            )

            # verify that all documents are found in the TSV file
            expected_doc_ids = [
                os.path.basename(f).replace(".tsv", "") for f in document_paths
            ]

            assert is_tsv_complete(dataset_path, expected_doc_ids)
            LOGGER.info(
                f"{dataset_path} contains all {len(expected_doc_ids)} expected documents"
            )



def create_dataset(
    files: str, language: str, split: str, version: str, output_dir: str
) -> str:
    """
    Creates one tsv file by concatenating all files in input
    :param files: tsv files with annotations
    :param language: language of files
    :param split: "train", "dev" or "test"
    :param version: version of data, will be included in filename
    :param output_dir: directory where tsv is to be stored

    :return: path where tsv file is stored
    """

    tsv_filename = f"newsagency-data-{version}-{split}-{language}.tsv"
    basedir = os.path.join(output_dir, version, language)

    if not os.path.exists(basedir):
        Path(basedir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(basedir, tsv_filename)
    concat_tsv_files(output_path, files)
    LOGGER.info(f"Written {split}-{language} to {output_path}")
    return output_path


def main(args):
    log_file = args["--log-file"]
    input_dir = args["--input-dir"]
    output_dir = args["--output-dir"]
    data_version = args["--data-version"]
    ann_planning = args["--annotation-planning"]
    discarded_dir = args["--discarded-dir"]
    language = args["--language"]

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    create_datasets(input_dir, output_dir, ann_planning, data_version, discarded_dir, language)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
