import argparse
import logging
import glob
from dask.diagnostics import ProgressBar
import dask.bag as db
from typing import Optional
import os
import sys
# The default value might be 1000
print(sys.getrecursionlimit())

import sys
sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000

# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)
# add the current directory to sys.path
sys.path.insert(0, current_directory)

from utils import Timer, chunk
logger = logging.getLogger(__name__)
import torch
import json
import pysbd
import string
import sys
from tqdm import tqdm

import requests

def send_prediction_request(json_request):
    url = 'http://127.0.0.1:8080/predictions/agency_fr'

    headers = {'Content-Type': 'application/json'}
    json_data = json.dumps(json_request)
    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code == 200:
        results = response.json()
    else:
        print(f'Failed to send prediction request. Error code: {response.status_code}')

    results = response.json()
    for result in results:
        if len(result) > 0:
            print(result)
            print('---'*20)
    return results


# TODO: add timing for 100 documents
def predict_entities(content_items):
    sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000
    import time
    # add the current directory to sys.path
    # current_directory = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, current_directory)

    timings = []

    torch.cuda.empty_cache()

    timings.append({'load_models': 0.0})

    SENTENCE_SEGMENTER = {'fr': pysbd.Segmenter(language="fr", clean=False),
                          'de': pysbd.Segmenter(language="de", clean=False)}

    count = 0
    result_json = []

    content_items_list = list(content_items)  # convert the filter object to a list

    for ci in tqdm(content_items_list, total=len(content_items_list)):
        count += 1
        timing = {}

        language = ci['lg_comp']
        article = ci['ft']
        if language in ['de', 'fr']:
            # TODO: add timing
            segmenter_start_time = time.time()
            sentences = SENTENCE_SEGMENTER[language].segment(article)
            if len(sentences) > 0:
                timing['segment'] = time.time() - segmenter_start_time  # Store the time taken
                cumulative_offset = 0

                article_prediction_start_time = time.time()
                timing_article = []

                api_requests = []
                for sentence in sentences:
                    api_requests.append(json.dumps({"text": sentence, "language": language}))

                timing_sentence = {}
                pred_start_time = time.time()
                article_entities = send_prediction_request(api_requests)
                timing_sentence['sentence_prediction'] = time.time() - pred_start_time  # Store the time taken

                json_start_time = time.time()
                for entities in article_entities:
                    for entity in entities:
                        if entity[1] != 'O':
                            if entity[0] not in string.punctuation:
                                if len(entity[0]) > 1:
                                    lOffset = entity[-1][0]
                                    rOffset = entity[-1][1]

                                    entity_json = {
                                        "entity": entity[1],
                                        "name": entity[0],
                                        "lOffset": lOffset,
                                        "rOffset": rOffset,
                                        'id': ci["id"] + f":{lOffset}:{rOffset}:newsag:bert_{language}"}

                                    result_json.append(entity_json)

                    timing_sentence['sentence_result_json'] = time.time() - json_start_time  # Store the time taken
                    # Update cumulative offset after processing each sentence
                    cumulative_offset += len(sentence) + 1
                    timing_article.append(timing_sentence)
                timing['entire_article'] = timing_article
            timings.append(timing)
            if count > 10:
                break
    # logger.info(f'Number of timings: {len(timings)}.')
    # logger.info(f'Number of entities: {len(result_json)}.')
    return result_json, timings


def run_newsagency_tagger(input_dir: str,
                          output_dir: str,
                          prefix: Optional[str] = None) -> None:
    import time

    t = Timer()
    total_time_start = time.time()
    timings = []

    if prefix is not None:
        path = f"{input_dir}/{prefix}*.jsonl.bz2"
    else:
        path = f"{input_dir}/*.jsonl.bz2"

    logger.info(f"Indexing files in {path}")
    file_time_start = time.time()
    files = glob.glob(path)
    logger.info(f'Number of files: {len(files)}. Time taken to read files: {time.time() - file_time_start}')

    batches = list(chunk(files, 10))
    total = len(batches)

    for i, b in enumerate(batches):
        batch_time_start = time.time()
        timing = {'batch': i}
        logger.info(f'Parsing {i}/{total}: {b}')

        read_time_start = time.time()
        bag_articles = db.read_text(b) \
            .filter(lambda s: len(s) > 2) \
            .map(json.loads) \
            .filter(lambda ci: ci['tp'] in ['ar', 'page'])

        logger.info(f'Time taken to read and filter articles: {time.time() - read_time_start}')
        timing['read_and_filter'] = time.time() - read_time_start

        process_time_start = time.time()
        with client:
            output = bag_articles.map_partitions(predict_entities).compute()
            bag_articles = output[::2]  # This takes every other element, starting from the first one
            timing_articles = output[1::2]  # This takes every other element, starting from the second one

            bag_articles = db.from_sequence(bag_articles)
            bag_mentions = bag_articles.map(json.dumps)

            timing['timing_articles'] = [json.dumps(article) for article in timing_articles]
            logger.info(f'Time taken to process articles: {time.time() - process_time_start}')
            timing['process_articles'] = time.time() - process_time_start

            write_time_start = time.time()
            with ProgressBar():
                bag_mentions.to_textfiles(f'{output_dir}/' + '*.jsonl.bz2')

        logger.info(f'Time taken to write mentions: {time.time() - write_time_start}')
        timing['write_articles'] = time.time() - write_time_start
        logger.info(f"Batch time: {t.tick()}. Total time taken for this batch: {time.time() - batch_time_start}")
        timing['batch_time'] = time.time() - batch_time_start

        timings.append(timing)
        with open('batch_timings_dask_ts_batch_8workers10batch.json', 'w') as file:
            json.dump(timings, file)

    client.close()

    logger.info(f'Total time taken for run_newsagency_tagger: {time.time() - total_time_start}')



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

    # client = Client('127.0.0.1:8000')
    n = torch.cuda.device_count()
    # Connect to an existing Dask scheduler
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)    # #
    # # # Or, start a local Dask cluster
    # client = Client(processes=False)

    # from dask_cuda import LocalCUDACluster
    #
    # with Client(processes=False) as client:
    #     with LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1,2,3") as cluster:
    #         client = Client(cluster)

    from dask.distributed import Worker

    run_newsagency_tagger(
        arguments.input_dir,
        arguments.output_dir,
        arguments.prefix)


