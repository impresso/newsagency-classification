#https://github.com/impresso/CLEF-HIPE-2020-internal/blob/master/lib/helpers/__init__.py

import re

from typing import NamedTuple
from stringdist import levenshtein_norm

import argparse
import io
import logging
import os
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Generator, List, Tuple

from cassis import Cas, load_cas_from_xmi, load_typesystem
from pycaprio import Pycaprio
from pycaprio.core.objects.project import Project
from pycaprio.mappings import InceptionFormat



LOGGER = logging.getLogger(__name__)

HYPHENS = ["-", "¬"]
PATTERN_HYPHEN_CLEANING = re.compile(fr'[{"".join(HYPHENS)}]\s*')


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