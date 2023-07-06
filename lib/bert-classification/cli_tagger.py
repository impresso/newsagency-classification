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
from tqdm import tqdm
from dask.distributed import Client
from utils import Timer, chunk

logger = logging.getLogger(__name__)
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from model import ModelForSequenceAndTokenClassification
import json
import os
import pysbd
import string
import re
import torch.nn.functional as F
from utils import set_seed, SEED

set_seed(2023)

class MultiSingletonModel:
    _instances = {}
    _models = {}
    _tokenizers = {}

    # @staticmethod
    def getInstance(language, model_path_dict, label_map_path_dict):
        if language not in MultiSingletonModel._instances:
            print(f'Instance {language} creating now')
            MultiSingletonModel(
                language,
                model_path_dict[language],
                label_map_path_dict[language])
        return MultiSingletonModel._models[language], MultiSingletonModel._tokenizers[language]

    def __init__(self, language, model_path, label_map_path):

        if language in MultiSingletonModel._instances:
            raise Exception(f"Model for language {language} is a singleton!")
        else:

            # Load the tokenizer
            MultiSingletonModel._tokenizers[language] = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True)

            traced_model_path = f"traced_model_{language}.pt"

            if os.path.exists(traced_model_path):
                try:
                    MultiSingletonModel._models[language] = torch.jit.load(
                        traced_model_path)
                except Exception:
                    MultiSingletonModel._models[language] = None
            else:
                MultiSingletonModel._models[language] = None

            if MultiSingletonModel._models[language] is None:
                config = AutoConfig.from_pretrained(
                    model_path,
                    problem_type="single_label_classification",
                    local_files_only=True)

                MultiSingletonModel._models[language] = ModelForSequenceAndTokenClassification.from_pretrained(
                    model_path,
                    config=config,
                    num_sequence_labels=2,
                    num_token_labels=len(label_map_path),
                    local_files_only=True)

                MultiSingletonModel._models[language] = MultiSingletonModel._models[language].to(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                scripted_model = torch.jit.trace(
                    MultiSingletonModel._models[language], [
                        torch.zeros(
                            (1, 1), dtype=torch.long).to(
                            'cuda' if torch.cuda.is_available() else 'cpu')], strict=False)
                torch.jit.save(scripted_model, traced_model_path)

                MultiSingletonModel._models[language] = torch.jit.load(
                    traced_model_path)
                MultiSingletonModel._models[language] = MultiSingletonModel._models[language].to(
                    'cuda' if torch.cuda.is_available() else 'cpu')

                MultiSingletonModel._models[language].eval()

            MultiSingletonModel._instances[language] = self





def tokenize(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ' + punctuation + ' ')
    return text.split()

    {'id': 'IMP-1978-08-10-a-i0084',
     'pp': [5],
     'd': '1978-08-10',
     'ts': '2019-10-10T08:02:03Z',
     'tp': 'ar',
     't': 'Amélioration de la lutte contre les maladies dans le canton de Berne',
     'ft': "Amélioration de la lutte contre les maladies dans le canton de Berne Le gouvernement bernois "
     "présentera au Grand Conseil une loi créant un fonds pour la lutte contre les maladies dans le canton de Berne. "
     "Ce nouveau texte législatif doit remplacer la loi portant création de ressources financières pour lutter contre "
     "la tuberculose, la poliomyélite, les affections rhumatismales et d'autres maladies de longue durée, loi qui a "
     "aujourd'hui plus de vingt ans. La nouvelle loi doit tenir _comnte de l'évolution médicale et juridique survenue "
     "depuis lors Elle a notamment pour but de développer la médecine préventive, précise l'Office d'information et de "
     "documentation du canton de Berne dans un communiqué. Le fonds fut créé en 1910 spécialement pour la tuberculose. Aujourd'hui il est employé pour lutter contre un certain nombre de maladies de longue durée. Un des motifs essentiels justifiant la révision de la loi est l'entrée en vigueur en 1974 d'une part de la loi bernoise sur les hôpitaux et d'autre part de la loi fédérale sur les épidémies. Lors de l'entrée en vigueur de la loi sur les hôpitaux et les écoles préparant aux professions hospitalières, le fonds fut libéré de certaines dépenses comme la participation aux constructions hospitalières. Dorénavant, en revanche, le fonds devra prendre en charge toutes les maladies énumérées dans la nouvelle loi et non plus seulement certaines d'entre elles, comme c'est le cas aujourd'hui. Les nouveautés contenues dans le projet de loi obligent à régler autrement l'alimentation du fonds. Ce dernier est une œuvre de solidarité de l'Etat et des communes comparable à celle fixant la répartition des charges entre les œuvres sociales et les hôpitaux. L'Etat et les communes seront notablement déchargés, surtout pendant les dix crémières années où les besoins financiers du fonds seront limités à peu près à vingt millions de francs. (ats) ",
     'lg_comp': 'fr'}


from tqdm import tqdm
def predict_entities(content_items):
    MODEL_PATHS = {
        'fr': '/scratch/newsagency-project/checkpoints/fr/checkpoint-4395',
        'de': '/scratch/newsagency-project/checkpoints/de/checkpoint-1752'}

    with open('data/label_map.json', "r") as f:
        label_map = json.load(f)
    reverted_label_map = {v: k for k, v in dict(label_map).items()}
    LABEL_MAPS = {"fr": reverted_label_map, "de": reverted_label_map}

    MODELS, TOKENIZERS = {}, {}
    for language in ['fr', 'de']:
        model, tokenizer = MultiSingletonModel.getInstance(
            language, MODEL_PATHS, LABEL_MAPS)
        MODELS[language] = model
        TOKENIZERS[language] = tokenizer

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    SENTENCE_SEGMENTER = {'fr': pysbd.Segmenter(language="fr", clean=False),
                          'de': pysbd.Segmenter(language="de", clean=False)}

    MAX_SEQ_LEN = 512


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

        language = ci['lg_comp']
        article = ci['ft']
        if language in ['de', 'fr']:
            sentences = SENTENCE_SEGMENTER[language].segment(article)

            cumulative_offset = 0

            for sentence in sentences:
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
                from nltk.chunk import conlltags2tree
                from nltk import pos_tag
                from nltk.tree import Tree
                def get_entities(tokens, tags):
                    tags = [tag.replace('S-', 'B-').replace('E-', 'I-') for tag in tags]
                    pos_tags = [pos for token, pos in pos_tag(tokens)]

                    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
                    ne_tree = conlltags2tree(conlltags)

                    entities = []
                    idx = 0
                    char_position = 0  # This will hold the current character position

                    for subtree in ne_tree:
                        # skipping 'O' tags
                        if type(subtree) == Tree:
                            original_label = subtree.label()
                            original_string = " ".join([token for token, pos in subtree.leaves()])

                            entity_start_position = char_position
                            entity_end_position = entity_start_position + len(original_string)

                            entities.append((original_string, original_label, (idx, idx + len(subtree)),
                                             (entity_start_position, entity_end_position)))
                            idx += len(subtree)

                            # Update the current character position
                            # We add the length of the original string + 1 (for the space)
                            char_position += len(original_string) + 1
                        else:
                            token, pos = subtree
                            # If it's not a named entity, we still need to update the character position
                            char_position += len(token) + 1  # We add 1 for the space
                            idx += 1

                    return entities

                def realign(text_sentence, out_label_preds):

                    preds_list, words_list, confidence_list = [], [], []
                    word_ids = TOKENIZERS[language](
                        text_sentence, is_split_into_words=True).word_ids()
                    for idx, word in enumerate(text_sentence):
                        beginning_index = word_ids.index(idx)
                        try:
                            preds_list.append(
                                reverted_label_map[out_label_preds[beginning_index]])
                        except Exception as ex:  # the sentence was longer then max_length
                            preds_list.append('O')
                        words_list.append(word)
                    return words_list, preds_list

                words_list, preds_list = realign(text_sentence, out_label_preds)
                # print(len(words_list), len(confidence_list))

                entities = get_entities(words_list, preds_list)
                # if len(entities) > 0: print('\n', entities)
                predicted_entities = list(zip(words_list, preds_list))

                for entity in entities:
                    if entity[1] != 'O':
                        if entity[0] not in string.punctuation:
                            if len(entity[0]) > 1:
                                lOffset = cumulative_offset + entity[-1][0]
                                rOffset = cumulative_offset + entity[-1][1]

                                entity_json = {
                                    "entity": entity[1],
                                    "name": entity[0],
                                    "lOffset": lOffset,
                                    "rOffset": rOffset,
                                    # "confidence": entity[2],
                                    'id': ci["id"] + f":{entity_json['lOffset']}:{entity_json['rOffset']}:newsag:bert_{language}"}

                                # print(entity_json)
                                # print(article[lOffset:rOffset], '----',  entity[0], '\n\n')
                                result_json.append(entity_json)

                # Update cumulative offset after processing each sentence
                cumulative_offset += len(sentence) + 1

    return result_json



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

    batches = list(chunk(files, 1))
    total = len(batches)
    for i, b in enumerate(batches):
        logger.info(f'Parsing {i}/{total}: {b}')
        bag_articles = db.read_text(b) \
            .filter(lambda s: len(s) > 2) \
            .map(json.loads) \
            .filter(lambda ci: ci['tp'] == 'ar')

        bag_mentions = bag_articles.map_partitions(
            predict_entities).map(json.dumps)

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

    # client = Client('127.0.0.1:8000')

    run_newsagency_tagger(
        arguments.input_dir,
        arguments.output_dir,
        arguments.prefix)
