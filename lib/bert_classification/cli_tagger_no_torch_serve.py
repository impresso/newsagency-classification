from tqdm import tqdm
import torch.nn.functional as F
import re
import string
import pysbd
import json
from transformers import AutoTokenizer, AutoConfig
import torch
import numpy as np
from nltk.tree import Tree
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from utils import Timer, chunk
import argparse
import logging
import glob
from dask.diagnostics import ProgressBar
import dask.bag as db
from typing import Optional

import os
import sys
from dask.distributed import Client
import sys

# The default value might be 1000
print(sys.getrecursionlimit())

sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000


# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)
# add the current directory to sys.path
sys.path.insert(0, current_directory)

logger = logging.getLogger(__name__)
# Get the directory of your script

print("Current directory:", os.getcwd())
print("Python path:", sys.path)

# set_seed(2023)


def load_models(model_paths, label_map_path):
    _instances = {}
    _models = {}
    _tokenizers = {}

    for language in ['fr', 'de']:

        current_directory = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(0, current_directory)

        from model import ModelForSequenceAndTokenClassification

        if language in _instances:
            raise Exception(f"Model for language {language} is a singleton!")
        else:

            # Load the tokenizer
            _tokenizers[language] = AutoTokenizer.from_pretrained(
                model_paths[language], local_files_only=True)

            traced_model_path = f"/scratch/newsagency-project/checkpoints/traced_model_{language}.pt"

            if os.path.exists(traced_model_path):
                try:
                    _models[language] = torch.jit.load(
                        traced_model_path)

                    logger.info(f'torch.jit.load: {language}')

                except Exception:
                    _models[language] = None
            else:
                _models[language] = None

            if _models[language] is None:
                config = AutoConfig.from_pretrained(
                    model_paths[language],
                    problem_type="single_label_classification",
                    local_files_only=True)

                _models[language] = ModelForSequenceAndTokenClassification.from_pretrained(
                    model_paths[language],
                    config=config,
                    num_sequence_labels=2,
                    num_token_labels=len(label_map_path),
                    local_files_only=True)

                _models[language] = _models[language].to(
                    'cuda' if torch.cuda.is_available() else 'cpu')

                scripted_model = torch.jit.trace(
                    _models[language], [
                        torch.zeros(
                            (1, 1), dtype=torch.long).to(
                            'cuda' if torch.cuda.is_available() else 'cpu')], strict=False)
                torch.jit.save(scripted_model, traced_model_path)

                _models[language] = torch.jit.load(
                    traced_model_path)
                _models[language] = _models[language].to(
                    'cuda' if torch.cuda.is_available() else 'cpu')

                _models[language].eval()

    return _models, _tokenizers


MODELS, TOKENIZERS = None, None


def load_models_once(model_paths, reverted_label_map):
    global MODELS, TOKENIZERS
    if MODELS is None or TOKENIZERS is None:
        MODELS, TOKENIZERS = load_models(model_paths, reverted_label_map)
    return MODELS, TOKENIZERS


def tokenize(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ' + punctuation + ' ')
    return text.split()


def get_entities(tokens, tags):
    tags = [tag.replace('S-', 'B-').replace('E-', 'I-') for tag in tags]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg)
                 for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    entities = []
    idx = 0
    char_position = 0  # This will hold the current character position

    for subtree in ne_tree:
        # skipping 'O' tags
        if isinstance(subtree, Tree):
            original_label = subtree.label()
            original_string = " ".join(
                [token for token, pos in subtree.leaves()])

            entity_start_position = char_position
            entity_end_position = entity_start_position + len(original_string)

            entities.append(
                (original_string,
                 original_label,
                 (idx,
                  idx + len(subtree)),
                    (entity_start_position,
                     entity_end_position)))
            idx += len(subtree)

            # Update the current character position
            # We add the length of the original string + 1 (for the space)
            char_position += len(original_string) + 1
        else:
            token, pos = subtree
            # If it's not a named entity, we still need to update the character
            # position
            char_position += len(token) + 1  # We add 1 for the space
            idx += 1

    return entities


def realign(
        text_sentence,
        out_label_preds,
        TOKENIZERS,
        language,
        reverted_label_map):
    preds_list, words_list, confidence_list = [], [], []
    word_ids = TOKENIZERS[language](
        text_sentence, is_split_into_words=True).word_ids()
    for idx, word in enumerate(text_sentence):

        try:
            beginning_index = word_ids.index(idx)
            preds_list.append(
                reverted_label_map[out_label_preds[beginning_index]])
        except Exception as ex:  # the sentence was longer then max_length
            preds_list.append('O')
        words_list.append(word)
    return words_list, preds_list


# TODO: add timing for 100 documents
def predict_entities(content_items):
    sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000
    import time
    # add the current directory to sys.path
    # current_directory = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, current_directory)

    timings = []

    torch.cuda.empty_cache()

    MODEL_PATHS = {
        'fr': '/scratch/newsagency-project/checkpoints/fr/checkpoint-4395',
        'de': '/scratch/newsagency-project/checkpoints/de/checkpoint-1752'}

    label_map = {
        "B-org.ent.pressagency.Reuters": 0,
        "B-org.ent.pressagency.Stefani": 1,
        "O": 2,
        "B-org.ent.pressagency.Extel": 3,
        "B-org.ent.pressagency.Havas": 4,
        "I-org.ent.pressagency.Xinhua": 5,
        "I-org.ent.pressagency.Domei": 6,
        "B-org.ent.pressagency.Belga": 7,
        "B-org.ent.pressagency.CTK": 8,
        "B-org.ent.pressagency.ANSA": 9,
        "B-org.ent.pressagency.DNB": 10,
        "B-org.ent.pressagency.Domei": 11,
        "I-pers.ind.articleauthor": 12,
        "I-org.ent.pressagency.Wolff": 13,
        "B-org.ent.pressagency.unk": 14,
        "I-org.ent.pressagency.Stefani": 15,
        "I-org.ent.pressagency.AFP": 16,
        "B-org.ent.pressagency.UP-UPI": 17,
        "I-org.ent.pressagency.ATS-SDA": 18,
        "I-org.ent.pressagency.unk": 19,
        "B-org.ent.pressagency.DPA": 20,
        "B-org.ent.pressagency.AFP": 21,
        "I-org.ent.pressagency.DNB": 22,
        "B-pers.ind.articleauthor": 23,
        "I-org.ent.pressagency.UP-UPI": 24,
        "B-org.ent.pressagency.Kipa": 25,
        "B-org.ent.pressagency.Wolff": 26,
        "B-org.ent.pressagency.ag": 27,
        "I-org.ent.pressagency.Extel": 28,
        "I-org.ent.pressagency.ag": 29,
        "B-org.ent.pressagency.ATS-SDA": 30,
        "I-org.ent.pressagency.Havas": 31,
        "I-org.ent.pressagency.Reuters": 32,
        "B-org.ent.pressagency.Xinhua": 33,
        "B-org.ent.pressagency.AP": 34,
        "B-org.ent.pressagency.APA": 35,
        "I-org.ent.pressagency.ANSA": 36,
        "B-org.ent.pressagency.DDP-DAPD": 37,
        "I-org.ent.pressagency.TASS": 38,
        "I-org.ent.pressagency.AP": 39,
        "B-org.ent.pressagency.TASS": 40,
        "B-org.ent.pressagency.Europapress": 41,
        "B-org.ent.pressagency.SPK-SMP": 42}

    reverted_label_map = {v: k for k, v in dict(label_map).items()}

    load_model_start_time = time.time()
    global MODELS, TOKENIZERS

    load_models_once(MODEL_PATHS, reverted_label_map)
    # Store the time taken
    timings.append({'load_models': time.time() - load_model_start_time})

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    SENTENCE_SEGMENTER = {'fr': pysbd.Segmenter(language="fr", clean=False),
                          'de': pysbd.Segmenter(language="de", clean=False)}

    MAX_SEQ_LEN = 512
    count = 0
    # for ci in content_items:
    #     count += 1
    # print('\nNumber of content_items', count)
    # total = count
    result_json = []

    # convert the filter object to a list
    content_items_list = list(content_items)

    for ci in tqdm(content_items_list, total=len(content_items_list)):
        count += 1

        timing = {}

        language = ci['lg_comp']
        article = ci['ft']
        if language in ['de', 'fr']:
            # TODO: add timing
            segmenter_start_time = time.time()
            sentences = SENTENCE_SEGMENTER[language].segment(article)
            # Store the time taken
            timing['segment'] = time.time() - segmenter_start_time
            cumulative_offset = 0

            article_prediction_start_time = time.time()
            timing_article = []
            for sentence in sentences:
                timing_sentence = {}
                # TODO: add timing
                tokenize_start_time = time.time()
                text_sentence = tokenize(sentence)

                tokenized_inputs = TOKENIZERS[language](
                    sentence,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    # We use this argument because the texts in our dataset are lists
                    # of words (with a label for each word).
                    # is_split_into_words=True
                )
                input_ids = torch.tensor([tokenized_inputs['input_ids']]).to(
                    'cuda' if torch.cuda.is_available() else 'cpu')

                timing_sentence['tokenize_and_tensor'] = time.time(
                ) - tokenize_start_time  # Store the time taken

                pred_start_time = time.time()
                with torch.no_grad():
                    outputs = MODELS[language](input_ids)

                sequence_result, tokens_result = outputs[0], outputs[1]

                token_logits = tokens_result['logits']
                out_token_preds = token_logits.detach().cpu().numpy()
                out_label_preds = np.argmax(out_token_preds, axis=2)[0]

                # apply softmax to convert logits to probabilities
                # probabilities = F.softmax(token_logits, dim=-1)
                # confidence_scores = probabilities.max(dim=-1)[0].cpu().numpy()

                # print(confidence_scores, len(confidence_scores[0]), len(out_label_preds))
                timing_sentence['sentence_prediction'] = time.time(
                ) - pred_start_time  # Store the time taken

                realign_start_time = time.time()
                words_list, preds_list = realign(
                    text_sentence, out_label_preds, TOKENIZERS, language, reverted_label_map)
                timing_sentence['realign'] = time.time(
                ) - realign_start_time  # Store the time taken

                get_entities_start_time = time.time()
                entities = get_entities(words_list, preds_list)
                timing_sentence['sentence_get_entities'] = time.time(
                ) - get_entities_start_time  # Store the time taken
                # logging.info(str(entities))
                # logging.info(type(entities))
                json_start_time = time.time()
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
                                    # "confidence": entity[2],
                                    'id': ci["id"] + f":{lOffset}:{rOffset}:newsag:bert_{language}"}

                                # print(entity_json)
                                # print(article[lOffset:rOffset], '----',  entity[0], '\n\n')
                                result_json.append(entity_json)
                timing_sentence['sentence_result_json'] = time.time(
                ) - json_start_time  # Store the time taken
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
    logger.info(
        f'Number of files: {len(files)}. Time taken to read files: {time.time() - file_time_start}')

    batches = list(chunk(files, 1))
    total = len(batches)

    global MODELS, TOKENIZERS

    for i, b in enumerate(batches):
        batch_time_start = time.time()
        timing = {'batch': i}
        logger.info(f'Parsing {i}/{total}: {b}')

        read_time_start = time.time()
        bag_articles = db.read_text(b) \
            .filter(lambda s: len(s) > 2) \
            .map(json.loads) \
            .filter(lambda ci: ci['tp'] == 'ar')
        logger.info(
            f'Time taken to read and filter articles: {time.time() - read_time_start}')
        timing['read_and_filter'] = time.time() - read_time_start

        process_time_start = time.time()
        with client:
            output = bag_articles.map_partitions(predict_entities).compute()
            # This takes every other element, starting from the first one
            bag_articles = output[::2]
            # This takes every other element, starting from the second one
            timing_articles = output[1::2]

            bag_articles = db.from_sequence(bag_articles)
            bag_mentions = bag_articles.map(json.dumps)

            timing['timing_articles'] = [
                json.dumps(article) for article in timing_articles]
            logger.info(
                f'Time taken to process articles: {time.time() - process_time_start}')
            timing['process_articles'] = time.time() - process_time_start

            write_time_start = time.time()
            with ProgressBar():
                bag_mentions.to_textfiles(f'{output_dir}/' + '*.jsonl.bz2')

        logger.info(
            f'Time taken to write mentions: {time.time() - write_time_start}')
        timing['write_articles'] = time.time() - write_time_start
        logger.info(
            f"Batch time: {t.tick()}. Total time taken for this batch: {time.time() - batch_time_start}")
        timing['batch_time'] = time.time() - batch_time_start

        timings.append(timing)
        with open('data/timinigs/batch_timings_dask.json', 'w') as file:
            json.dump(timings, file)

    client.close()

    logger.info(
        f'Total time taken for run_newsagency_tagger: {time.time() - total_time_start}')


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

    cluster = LocalCluster(n_workers=2)
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
