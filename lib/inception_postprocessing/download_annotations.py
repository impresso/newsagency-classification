"""
Script to download impresso-newsagency-project annotated documents from INCEpTION,
largely based on HIPE code (https://github.com/impresso/CLEF-HIPE-2020-internal/blob/master/lib/download_annotations.py).

NB: to be able to run this script you need to be a user who has the rights to make API calls to an INCEpTION instance.

Usage:
    download_annotations.py --language=<l> --output-dir=<od> --log-file=<log> --annotation-planning=<ann>
"""  # noqa

# Set the following environment variables to access Inception
# INCEPTION_HOST=https://inception.dhlab.epfl.ch/prod/
# INCEPTION_USERNAME=*
# INCEPTION_PASSWORD=*


from __future__ import print_function

import ipdb as pdb
import logging
import os.path
from pathlib import Path

import pandas as pd
from docopt import docopt
from tqdm import tqdm

from helpers import (
    download_document,
    find_project_by_name,
    index_project_documents,
    make_inception_client,
)


def run_download_annotations(
    lang: str, output_dir: str, ann_planning: str
) -> None:
    """Downloads annotated documents from INCEpTION.

    This function relies on the fact that HIPE project names in INCEpTION comply to the following template:
    `impresso-newsagencies: {lang}`. Also, the script uses an external spreadsheet to determine whose annotations
    are to be trusted as the final version of the annotated document.

    :param str lang: Language of annotated documents.
    :param str output_dir: Path of output directory where to download annotated documents.
    :param str ann_planning: Path to csv file with annotation planning (1 row per doc)
    :return: Does not return anything.
    :rtype: None

    """

    # create the inception pycaprio client
    inception_client = make_inception_client()

    # read annotation assignments for all annotated documents
    assignees_df = pd.read_csv(
                ann_planning
        )
    assignees_df = assignees_df[assignees_df["Corpus"] == lang]

    # create a couple of inverted indexes to be able to roundtrip from document name (canonical) to
    # inception document ID and viceversa
    inception_project = find_project_by_name(
        inception_client, f"impresso-newsagencies_-{lang.lower()}"
    )
    idx_id2name, idx_name2id = index_project_documents(
        inception_project.project_id, inception_client
    )

    # create the download folder if not yet existing
    download_dir = os.path.join(output_dir, lang)
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    #only download finished annotations
    finished_df = assignees_df[assignees_df["Finished Annotation"] == True]

    logging.info(f"There are {finished_df.shape[0]} annotated documents; ")

    for index, row in tqdm(list(finished_df.iterrows())):

        assignee = row["Annotator"]
        document_name = row["Document ID"] + ".xmi"

        
        try:
            assert document_name in idx_name2id
        except AssertionError:
            logging.error(f"{document_name} not found in {inception_project}")
            continue

        document_inception_id = idx_name2id[document_name]
        #print(document_inception_id, document_name)
        
        download_document(
            document_id=document_inception_id,
            project_id=inception_project.project_id,
            output_dir=download_dir,
            user=assignee,
            inception_client=inception_client,
        )


def main(args):

    lang = args["--language"]
    output_dir = args["--output-dir"]
    log_file = args["--log-file"]
    ann_planning = args["--annotation-planning"]

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_download_annotations(lang, output_dir, ann_planning)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)