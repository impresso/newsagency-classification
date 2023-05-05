import re

from typing import NamedTuple
from stringdist import levenshtein_norm
from os import listdir
from os.path import isfile, join

import argparse
import io
import logging
import os
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Generator, List, Tuple
import pandas as pd

from cassis import Cas, load_cas_from_xmi, load_typesystem
from pycaprio import Pycaprio
from pycaprio.core.objects.project import Project
from pycaprio.mappings import InceptionFormat

ANNOTATION_PLANNING = "Annotation Planning.csv"
DATA_PATH = "data/"


LOGGER = logging.getLogger(__name__)

HYPHENS = ["-", "¬"]
PATTERN_HYPHEN_CLEANING = re.compile(fr'[{"".join(HYPHENS)}]\s*')




def get_annotated_ids(annotated_docs: str, lang: str) -> List[str]:
    """ 
    Parameters:
        annotated_docs: either directory which contains all the articles for annotation (xmi-articles are queried in subfolder core_{lang}/xmi/)
                        or path of txt-file which contains list of all the annotated articles
        lang: language ("de" or "fr")
    """
    if os.path.isfile(annotated_docs):
        with open(annotated_docs, "r") as f:
            doc_IDs = [line.rstrip() for line in f]

    if os.path.isdir(annotated_docs):
        annotation_core_dir = join(DATA_PATH, f"for_annotation/core_{lang.lower()}/xmi/")
        annotation_IA_dir = join(DATA_PATH, f"for_annotation/IA_{lang.lower()}/xmi/")

        #get document IDs of docs which were annotated for the project 
        doc_IDs_with_ending = [f for f in listdir(annotation_core_dir) if isfile(join(annotation_core_dir, f))]
        doc_IDs_with_ending += [f for f in listdir(annotation_IA_dir) if isfile(join(annotation_IA_dir, f))]

        doc_IDs = sorted([filename[:-4] for filename in doc_IDs_with_ending])

        with open(annotated_docs + f"annotated_{lang}_docIDs.txt", "w") as f:
            for id in doc_IDs:
                f.write(id +"\n")
    
    return doc_IDs
        


def annotation_planning2df(annotation_planning_path: str):
    """
    Takes the csv file used to organise for annotation and turns it into a pandas Dataframe with the following columns:
        Annotator (str), Inception Project (str), Newspapers (list of str), Finished Annotation (bool)
    :rtype pd.DataFrame
    """
    ann_planning = pd.read_csv(annotation_planning_path, usecols=["Annotator", "Inception Project", "Newspapers", "Finished Annotation"])
    ann_planning['Annotator'] = ann_planning['Annotator'].fillna(method='ffill')
    ann_planning['Newspapers'] = ann_planning['Newspapers'].apply(lambda x: x.split(", "))

    return ann_planning


def make_annotation_planning_per_doc(annotation_dir: str, lang: str):
    """
    Makes an annotation planning for one project, with one row per annotated document

    Parameters:
        annotation_dir: path to the directory containing the annotation planning (1 row per project & annotator), 
                                    as well as a txt-file with all the articles which were annotated
        lang: language ("de" or "fr")
        output_dir: directory where the "new" annotation planning should be stored

    returns: DataFrame with columns ['Document ID', 'Annotator', 'Inception Project', 'Finished Annotation']
    :rtype pd.DataFrame
    """

    #specify the project and its directory
    project = "impresso-newsagencies: " + lang.upper()
  

    #get the annotation planning and select only the lines specific to the project
    annotation_df = annotation_planning2df(annotation_dir + ANNOTATION_PLANNING)
    annotation_df = annotation_df.loc[annotation_df['Inception Project'] == project]

    #get IDs of all annotated documents
    doc_IDs = get_annotated_ids(annotation_dir, lang)

    #join annotation planning with document IDs (-> one entry per doc ID)
    annotation_df = annotation_df.explode("Newspapers")
    doc_IDs_df = pd.DataFrame(doc_IDs, columns=['Document ID'])
    doc_IDs_df['Newspapers'] = doc_IDs_df['Document ID'].apply(lambda s: s.split("-")[0])
    annotation_df = doc_IDs_df.merge(annotation_df, on='Newspapers')
    annotation_df.drop("Newspapers", axis=1, inplace=True)
    
    return annotation_df



"""
from https://github.com/impresso/CLEF-HIPE-2020-internal/blob/master/lib/helpers/__init__.py
"""


ImpressoDocument = NamedTuple(
    "ImpressoDocument",
    [
        ("newspaper", str),
        ("date", str),
        ("id", str),
        ("filename", str),
        ("filepath", str),
        ("segments", dict),
        ("autosentences", dict),
        ("mentions", dict),
        ("links", list),
        ("relations", list),
        ("text", str),
    ],
)


def compute_levenshtein_distance(surface: str, transcript: str) -> int:
    """Compute the normalized Levensthein distance between two strings after cleaning
    :param str surface: a reference string.
    :param str transcript: a candidate string.
    :return: Levensthein distance
    :rtype: int
    """

    def clean(text: str) -> str:
        """
        Remove the symbols "-" or "¬" together with potential whitespace which may follow
        """
        return PATTERN_HYPHEN_CLEANING.sub("", text,)

    return levenshtein_norm(clean(surface), clean(transcript))


def is_tsv_complete(dataset_path: str, expected_doc_ids: List[str]) -> bool:

    with open(dataset_path, "r", encoding="utf-8") as f:
        tsv_doc_ids = [
            line.strip().split("=")[-1].strip()
            for line in f.readlines()
            if line.startswith("#") and "document_id" in line
        ]

    difference = set(expected_doc_ids).difference(set(tsv_doc_ids))

    try:
        assert difference == set()
        return True
    except AssertionError:
        print(f"Following documents are missing from {dataset_path}: {difference}")
        return False




"""
from https://github.com/impresso/CLEF-HIPE-2020-internal/tree/master/lib/helpers
"""


def find_project_by_name(inception_client: Pycaprio, project_name: str) -> Project:
    """Finds a project in INCEpTION by its name.

    :param Pycaprio inception_client: Pycaprio INCEpTION client.
    :param str project_name: project name.
    :return: Description of returned object.
    :rtype: type

    """
    matching_projects = [project for project in inception_client.api.projects() if project.project_name == project_name]
    assert len(matching_projects) == 1
    return matching_projects[0]



def make_inception_client():
    """Creates a Pycaprio client to INCEpTION.

    Connection parameters are read from environment variables.
    """
    LOGGER.info(
        (
            'Using following INCEpTION connection parameters (from environment): '
            f"Host: {os.environ['INCEPTION_HOST']}; user: {os.environ['INCEPTION_USERNAME']}; "
            f"password: {'set (hidden for security)' if 'INCEPTION_PASSWORD' in os.environ else 'not set'}"
        )
    )
    return Pycaprio()


def index_project_documents(project_id: int, inception_client: Pycaprio) -> Tuple[dict, dict]:
    """Creates two inverted indexes of documents in a project in INCEpTION.

    :param int project_id: Description of parameter `project_id`.
    :param Pycaprio inception_client: Description of parameter `inception_client`.
    :return: Description of returned object.
    :rtype: Tuple[dict, dict]

    """
    id2name_idx = {}
    name2id_idx = {}
    documents = inception_client.api.documents(project_id)
    for document in documents:
        id2name_idx[document.document_id] = document.document_name
        name2id_idx[document.document_name] = document.document_id
    return (id2name_idx, name2id_idx)


def download_document(
    project_id: int, document_id: int, user: str, output_dir: str, inception_client: Pycaprio
) -> None:
    """Downloads an annotated document (by ID) from INCEpTION as unzipped UIMA/XMI."""

    annotated_doc = inception_client.api.annotation(
        project_id, document_id, user, annotation_format=InceptionFormat.UIMA_CAS_XMI
    )
    
    xmi_document = zipfile.ZipFile(io.BytesIO(annotated_doc))
    xmi_document.extractall(output_dir)
    LOGGER.info(f'Downloaded annotations from user {user} for {document_id} from project {project_id}')