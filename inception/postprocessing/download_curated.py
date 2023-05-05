#!/usr/bin/env python
# coding: utf-8

# TODO: check if the file is already in the output directory

"""
CLI script to download curated documents from inception.

Usage:
    download_curated.py --user=<u> --password=<pwd> --project-id=<pid> --output-dir=<od> --api-endpoint=<api> [--name-contains=<name>]
"""

import requests
import zipfile
import io
from requests.auth import HTTPBasicAuth
from docopt import docopt

__author__ = "Matteo Romanello"
__email__ = "matteo.romanello@epfl.ch"
__organisation__ = "DH Lab, EPFL"
__status__ = "development"


def fetch_documents(project_id, username, password, api_endpoint):

    req_uri = f'{api_endpoint}projects/{project_id}/documents'
    print(req_uri)
    authentication = HTTPBasicAuth(username, password)
    return requests.get(req_uri, auth=authentication).json()['body']


def download_curated_document(project_id, document_id, username, password, download_path, api_endpoint):
    try:
        req_uri = f'{api_endpoint}projects/{project_id}/documents/{document_id}/curation'
        authentication = HTTPBasicAuth(username, password)
        r = requests.get(req_uri, auth=authentication, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        filenames = [f.filename for f in z.filelist]
        print(f"[proj={project_id}, doc={document_id}]", f"Following files were downloaded: {','.join(filenames)}")
        z.extractall(path=download_path)
        return True
    except Exception as e:
        print(e)
        return False


def main(args):

    project_id = args['--project-id']
    user = args['--user']
    pwd = args['--password']
    out_dir = args['--output-dir']
    name_filter = args['--name-contains']
    api_endpoint = args['--api-endpoint']

    not_completed = []
    for doc in fetch_documents(project_id, user, pwd, api_endpoint):

        if name_filter is not None:
            if name_filter not in doc['name']:
                continue

        if doc['state'] == 'CURATION-COMPLETE':
            print(f"Doc {doc['id']} {doc['name']} is" f"{doc['state']} and will be downloaded")
            success = download_curated_document(project_id, doc['id'], user, pwd, out_dir, api_endpoint)
            assert success

            # TODO: rename stage1 to stage2
        else:
            not_completed.append(doc['name'])
    if not_completed:
        print("The following files have not been downloaded (curation not completed yet):\n", not_completed)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
