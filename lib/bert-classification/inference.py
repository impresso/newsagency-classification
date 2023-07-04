# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from model import ModelForSequenceAndTokenClassification
import json
import os
import pysbd
import string

from transformers import AutoTokenizer


class MultiSingletonModel:
    _instances = {}
    _models = {}
    _tokenizers = {}

    @staticmethod
    def getInstance(language, model_path_dict, label_map_path_dict):
        if language not in MultiSingletonModel._instances:
            print(f'Instance {language} creating now')
            MultiSingletonModel(language, model_path_dict[language], label_map_path_dict[language])
        return MultiSingletonModel._models[language], MultiSingletonModel._tokenizers[language]

    def __init__(self, language, model_path, label_map_path):

        if language in MultiSingletonModel._instances:
            raise Exception(f"Model for language {language} is a singleton!")
        else:

            # Load the tokenizer
            MultiSingletonModel._tokenizers[language] = AutoTokenizer.from_pretrained(model_path,
                                                                                      local_files_only=True)

            traced_model_path = f"traced_model_{language}.pt"

            if os.path.exists(traced_model_path):
                try:
                    MultiSingletonModel._models[language] = torch.jit.load(traced_model_path)
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
                scripted_model = torch.jit.trace(MultiSingletonModel._models[language], [
                    torch.zeros((1, 1), dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')],
                                                 strict=False)
                torch.jit.save(scripted_model, traced_model_path)

                MultiSingletonModel._models[language] = torch.jit.load(traced_model_path)

                MultiSingletonModel._models[language].eval()

            MultiSingletonModel._instances[language] = self


MODEL_PATHS = {'fr': '/scratch/newsagency-project/checkpoints/fr/checkpoint-4395',
               'de': '/scratch/newsagency-project/checkpoints/de/checkpoint-1752'}

with open('data/label_map.json', "r") as f:
    label_map = json.load(f)
reverted_label_map = {v: k for k, v in dict(label_map).items()}
LABEL_MAPS = {"fr": reverted_label_map, "de": reverted_label_map}

MODELS, TOKENIZERS = {}, {}
for language in ['fr', 'de']:
    model, tokenizer = MultiSingletonModel.getInstance(language, MODEL_PATHS, LABEL_MAPS)
    MODELS[language] = model
    TOKENIZERS[language] = tokenizer

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
SENTENCE_SEGMENTER = {'fr': pysbd.Segmenter(language="fr", clean=False),
                      'de': pysbd.Segmenter(language="de", clean=False)}

MAX_SEQ_LEN = 512
def tokenize(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ' + punctuation + ' ')
    return text.split()

    {'id': 'IMP-1978-08-10-a-i0084', 'pp': [5], 'd': '1978-08-10', 'ts': '2019-10-10T08:02:03Z', 'tp': 'ar',
     't': 'Amélioration de la lutte contre les maladies dans le canton de Berne',
     'ft': "Amélioration de la lutte contre les maladies dans le canton de Berne Le gouvernement bernois "
           "présentera au Grand Conseil une loi créant un fonds pour la lutte contre les maladies dans le canton de Berne. "
           "Ce nouveau texte législatif doit remplacer la loi portant création de ressources financières pour lutter contre "
           "la tuberculose, la poliomyélite, les affections rhumatismales et d'autres maladies de longue durée, loi qui a "
           "aujourd'hui plus de vingt ans. La nouvelle loi doit tenir _comnte de l'évolution médicale et juridique survenue "
           "depuis lors Elle a notamment pour but de développer la médecine préventive, précise l'Office d'information et de "
           "documentation du canton de Berne dans un communiqué. Le fonds fut créé en 1910 spécialement pour la tuberculose. Aujourd'hui il est employé pour lutter contre un certain nombre de maladies de longue durée. Un des motifs essentiels justifiant la révision de la loi est l'entrée en vigueur en 1974 d'une part de la loi bernoise sur les hôpitaux et d'autre part de la loi fédérale sur les épidémies. Lors de l'entrée en vigueur de la loi sur les hôpitaux et les écoles préparant aux professions hospitalières, le fonds fut libéré de certaines dépenses comme la participation aux constructions hospitalières. Dorénavant, en revanche, le fonds devra prendre en charge toutes les maladies énumérées dans la nouvelle loi et non plus seulement certaines d'entre elles, comme c'est le cas aujourd'hui. Les nouveautés contenues dans le projet de loi obligent à régler autrement l'alimentation du fonds. Ce dernier est une œuvre de solidarité de l'Etat et des communes comparable à celle fixant la répartition des charges entre les œuvres sociales et les hôpitaux. L'Etat et les communes seront notablement déchargés, surtout pendant les dix crémières années où les besoins financiers du fonds seront limités à peu près à vingt millions de francs. (ats) ",
     'lg_comp': 'fr'}

def predict_entities(content_items):
    result_json = []
    for ci in content_items:
        entity_json = {
            "entity": "newsag",
            "name": "Reuters",
            "lOffset": 2637,
            "rOffset": 2642,
            "id": ci["id"] + ":2637:2642:newsag:bert"
        }
        # print(ci)

        language = ci['lg_comp']
        article = ci['ft']
        sentences = SENTENCE_SEGMENTER[language].segment(article)

        cumulative_offset = 0

        from tqdm import tqdm

        for sentence in tqdm(sentences, total=len(sentences)):
            text_sentence = tokenize(sentence)

            tokenized_inputs = TOKENIZERS[language](
                sentence,
                padding="max_length",
                truncation=True,
                max_length=512,
                # We use this argument because the texts in our dataset are lists
                # of words (with a label for each word).
                # is_split_into_words=True
            )
            input_ids = torch.tensor([tokenized_inputs['input_ids']]).to('cuda' if torch.cuda.is_available() else 'cpu')

            with torch.no_grad():
                outputs = MODELS[language](input_ids)

            sequence_result, tokens_result = outputs[0], outputs[1]

            token_logits = tokens_result['logits']
            out_token_preds = token_logits.detach().cpu().numpy()
            out_label_preds = np.argmax(out_token_preds, axis=2)[0]

            def realign(text_sentence, out_label_preds):

                preds_list = []
                words_list = []
                word_ids = TOKENIZERS[language](text_sentence, is_split_into_words=True).word_ids()
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
            predicted_entities = list(zip(words_list, preds_list))

            for entity in predicted_entities:
                if entity[1] != 'O':

                    lOffset = cumulative_offset + sentence.find(entity[0])
                    rOffset = lOffset + len(entity[0])

                    entity_json = {
                        "entity": entity[1],
                        "name": entity[0],
                        "lOffset": lOffset,
                        "rOffset": rOffset,
                        'id': ci["id"] + f":{entity_json['lOffset']}:{entity_json['rOffset']}:newsag:bert_{language}"}

                    print(entity_json)
                    result_json.append(entity_json)

    return result_json




if __name__ == '__main__':

    sent = "Reuter est une agence de presse détenue par Thomson Reuters Corporation. " \
           "Elle emploie environ 2 500 journalistes et 600 photojournalistes dans environ 200 sites dans le monde. " \
           "Reuter est l'une des plus grandes agences de presse au monde. L'agence a été créée à Londres en 1851 par l'Allemand Paul Reuter."
    sent2 =  "Havas Amélioration de la lutte contre les maladies dans le canton de Berne Le gouvernement bernois "+\
           "présentera au Grand Conseil une loi créant un fonds pour la lutte contre les maladies dans le canton de Berne. "+\
           "Ce nouveau texte législatif doit remplacer la loi portant création de ressources financières pour lutter contre "+\
           "la tuberculose, la poliomyélite, les affections rhumatismales et d'autres maladies de longue durée, loi qui a "+\
           "aujourd'hui plus de vingt ans. La nouvelle loi doit tenir _comnte de l'évolution médicale et juridique survenue "+\
           "depuis lors Elle a notamment pour but de développer la médecine préventive, précise l'Office d'information et de "+\
           "documentation du canton de Berne dans un communiqué. Le fonds fut créé en 1910 spécialement pour la tuberculose. Aujourd'hui il est employé pour lutter contre un certain nombre de maladies de longue durée. Un des motifs essentiels justifiant la révision de la loi est l'entrée en vigueur en 1974 d'une part de la loi bernoise sur les hôpitaux et d'autre part de la loi fédérale sur les épidémies. Lors de l'entrée en vigueur de la loi sur les hôpitaux et les écoles préparant aux professions hospitalières, le fonds fut libéré de certaines dépenses comme la participation aux constructions hospitalières. Dorénavant, en revanche, le fonds devra prendre en charge toutes les maladies énumérées dans la nouvelle loi et non plus seulement certaines d'entre elles, comme c'est le cas aujourd'hui. Les nouveautés contenues dans le projet de loi obligent à régler autrement l'alimentation du fonds. Ce dernier est une œuvre de solidarité de l'Etat et des communes comparable à celle fixant la répartition des charges entre les œuvres sociales et les hôpitaux. L'Etat et les communes seront notablement déchargés, surtout pendant les dix crémières années où les besoins financiers du fonds seront limités à peu près à vingt millions de francs. (ats) "
    content_items = [{'ft': sent}, {'ft': sent2}]

    predict_entities(content_items)