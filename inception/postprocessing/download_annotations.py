"""
Script to download impresso-newsagency-project annotated documents from INCEpTION,
largely based on HIPE code (https://github.com/impresso/CLEF-HIPE-2020-internal/blob/master/lib/download_annotations.py).

NB: to be able to run this script you need to be a user who has the rights to make API calls to an INCEpTION instance.

Usage:
    download_annotations.py --language=<l> --input-dir=<id> --output-dir=<od> --log-file=<log> --annotation-planning=<ann>
"""  # noqa

from __future__ import print_function

import ipdb as pdb
import logging
from os import listdir
from os.path import isfile, join
import os.path
from pathlib import Path

import pandas as pd
from docopt import docopt
from tqdm import tqdm

from helpers_inception import (
    download_document,
    find_project_by_name,
    index_project_documents,
    make_inception_client,
)



def annotation_planning2df(annotation_planning_path):
    """
    Takes the csv file used to organise for annotation and turns it into a pandas Dataframe with the following columns:
        Annotator (str), Inception Project (str), Newspapers (list of str), Finished Annotation (bool)
    """
    ann_planning = pd.read_csv(annotation_planning_path, usecols=["Annotator", "Inception Project", "Newspapers", "Finished Annotation"])
    ann_planning['Annotator'] = ann_planning['Annotator'].fillna(method='ffill')
    ann_planning['Newspapers'] = ann_planning['Newspapers'].apply(lambda x: x.split(", "))

    return ann_planning


def make_annotation_planning_per_doc(annotation_dir, annotation_planning_path, lang):
    """
    Makes an annotation planning for one project, with one row per annotated document

    Parameters:
        annotation_dir: directory which contains all the articles for annotation (xmi-articles are queried in subfolder core_{lang}/xmi/)
        annotation_planning_path: path to the document containing the annotation planning (1 row per project & annotator)
        lang: language ("de" or "fr")
        output_dir: directory where the "new" annotation planning should be stored
    """

    #specify the project and its directory
    project = "impresso-newsagencies: " + lang.upper()
    annotation_project_dir = annotation_dir + f"core_{lang.lower()}/xmi/"

    #get document IDs of docs which were annotated for the project 
    doc_IDs_with_ending = [f for f in listdir(annotation_project_dir) if isfile(join(annotation_project_dir, f))]
    doc_IDs = [filename[:-4] for filename in doc_IDs_with_ending]

    #get the annotation planning and select only the lines specific to the project
    annotation_df = annotation_planning2df(annotation_planning_path)
    annotation_df = annotation_df.loc[annotation_df['Inception Project'] == project]

    #join annotation planning with document IDs (-> one entry per doc ID)
    annotation_df = annotation_df.explode("Newspapers")
    doc_IDs_df = pd.DataFrame(doc_IDs, columns=['Document ID'])
    doc_IDs_df['Newspapers'] = doc_IDs_df['Document ID'].apply(lambda s: s.split("-")[0])
    annotation_df = doc_IDs_df.merge(annotation_df, on='Newspapers')
    annotation_df.drop("Newspapers", axis=1, inplace=True)

    return annotation_df



def run_download_annotations(
    lang: str, input_dir: str, output_dir: str, ann_planning: str
) -> None:
    """Downloads annotated documents from INCEpTION.

    This function relies on the fact that HIPE project names in INCEpTION comply to the following template:
    `impresso-newsagencies: {lang}`. Also, the script uses an external spreadsheet to determine whose annotations
    are to be trusted as the final version of the annotated document.

    :param str lang: Language of annotated documents.
    :param str input_dir: Path of input directory (with documents uploaded to Inception for annotation).
    :param str output_dir: Path of output directory where to download annotated documents.
    :param str ann_planning: Path of the annotation planning (csv with one row per project & annotator expected).
    :return: Does not return anything.
    :rtype: None

    """

    # create the inception pycaprio client
    inception_client = make_inception_client()

    # read annotation assignments for all annotated documents
    assignees_df = make_annotation_planning_per_doc(
                input_dir, ann_planning, lang
        )

    import pdb; pdb.set_trace()
    # create a couple of inverted indexes to be able to roundtrip from document name (canonical) to
    # inception document ID and viceversa
    inception_project = find_project_by_name(
        inception_client, f"impresso-newsagencies: {lang.upper()}"
    )
    idx_id2name, idx_name2id = index_project_documents(
        inception_project.project_id, inception_client
    )

    # create the download folder if not yet existing
    download_dir = os.path.join(output_dir, lang)
    Path(download_dir).mkdir(exist_ok=True)

    #only download finished annotations
    finished_df = assignees_df[assignees_df["Finished Annotation"] == True]

    logging.info(f"There are {finished_df.shape[0]} annotated documents; ")

    for index, row in tqdm(list(finished_df.iterrows())):

        assignee = row["Annotator"]
        document_name = row["Document ID"]

        try:
            assert document_name in idx_name2id
        except AssertionError:
            logging.error(f"{document_name} not found in {inception_project}")
            continue

        document_inception_id = idx_name2id[document_name]

        download_document(
            document_id=document_inception_id,
            project_id=inception_project.project_id,
            output_dir=download_dir,
            user=assignee,
            inception_client=inception_client,
        )


def main(args):

    lang = args["--language"]
    input_dir = args["--input-dir"]
    output_dir = args["--output-dir"]
    log_file = args["--log-file"]
    ann_planning = args["--annotation-planning"]

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_download_annotations(lang, input_dir, output_dir, ann_planning)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)