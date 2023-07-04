# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from model import ModelForSequenceAndTokenClassification
import json
import os
import pysbd
import string

MODEL_NAME = '/scratch/newsagency-project/checkpoints/fr/checkpoint-4395'


class SingletonModel:
    _instance = None
    _model = None

    @staticmethod
    def getInstance(model_path=MODEL_NAME, label_map_path='data/label_map.json'):
        if SingletonModel._instance == None:
            SingletonModel(model_path, label_map_path)
        return SingletonModel._instance

    def __init__(self, model_path, label_map_path):
        if SingletonModel._instance != None:
            raise Exception("This class is a singleton!")
        else:
            traced_model_path = "traced_model.pt"

            if os.path.exists(traced_model_path):
                try:
                    SingletonModel._model = torch.jit.load(traced_model_path)
                except Exception:
                    SingletonModel._model = None
            else:
                SingletonModel._model = None

            if SingletonModel._model is None:
                config = AutoConfig.from_pretrained(
                    model_path,
                    problem_type="single_label_classification",
                    local_files_only=True)

                SingletonModel._model = ModelForSequenceAndTokenClassification.from_pretrained(
                    model_path,
                    config=config,
                    num_sequence_labels=2,
                    num_token_labels=len(json.load(open(label_map_path, "r"))),
                    local_files_only=True)


                scripted_model = torch.jit.script(SingletonModel._model)
                SingletonModel._model = SingletonModel._model.to('cuda' if torch.cuda.is_available() else 'cpu')


                scripted_model = torch.jit.trace(SingletonModel._model, [torch.zeros((1, 1), dtype=torch.long).to('cuda')], strict=False)
                torch.jit.save(scripted_model, traced_model_path)

                SingletonModel._model = torch.jit.load(traced_model_path)

                SingletonModel._model.eval()

            SingletonModel._instance = self


singleton_model = SingletonModel.getInstance()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
seg = pysbd.Segmenter(language="fr", clean=False)
with open('data/label_map.json', "r") as f:
    label_map = json.load(f)

reverted_label_map = {v: k for k, v in dict(label_map).items()}

MAX_SEQ_LEN = 512
def tokenize(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ' + punctuation + ' ')
    return text.split()


def predict_entities(content_items):
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

    # import pdb;pdb.set_trace()

    for ci in content_items:
        entity_json = {
            "entity": "newsag",
            "name": "Reuters",
            "lOffset": 2637,
            "rOffset": 2642,
            # "id": ci["id"] + ":2637:2642:newsag:bert"
        }
        # print(ci)

        sentences = seg.segment(ci['ft'])

        result_json = {"entities": []}

        from tqdm import tqdm

        for sentence in tqdm(sentences, total=len(sentences)):
            text_sentence = tokenize(sentence)

            tokenized_inputs = tokenizer(
                sentence,
                padding="max_length",
                truncation=True,
                max_length=512,
                # We use this argument because the texts in our dataset are lists
                # of words (with a label for each word).
                # is_split_into_words=True
            )
            # print(input_ids)
            input_ids = torch.tensor([tokenized_inputs['input_ids']]).to('cuda' if torch.cuda.is_available() else 'cpu')
            # if len(input_ids) > MAX_SEQ_LEN:
            #     # Truncate the input ids if they are too long
            #     input_ids = input_ids[:MAX_SEQ_LEN]
            # input_ids = tokenizer.encode(sentence, add_special_tokens=True)
            # input_ids = torch.tensor([input_ids]).to('cuda' if torch.cuda.is_available() else 'cpu')

            with torch.no_grad():
                outputs = singleton_model._model(input_ids)

            sequence_result, tokens_result = outputs[0], outputs[1]

            token_logits = tokens_result['logits']
            out_token_preds = token_logits.detach().cpu().numpy()
            out_label_preds = np.argmax(out_token_preds, axis=2)[0]

            # the second return value is logits
            # sequence_logits = sequence_result.logits
            # print(predicted_indices, type(predicted_indices))

            # predicted_indices = torch.argmax(logits, dim=1)

            # out_label_ids = [reverted_label_map[index] for index in out_token_ids]

            # words = tokenizer.convert_ids_to_tokens(input_ids[0])

            def realign(text_sentence, out_label_preds):

                preds_list = []
                words_list = []
                word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
                for idx, word in enumerate(text_sentence):
                    beginning_index = word_ids.index(idx)
                    try:
                        preds_list.append(
                            reverted_label_map[out_label_preds[beginning_index]])
                    except Exception as ex:  # the sentence was longer then max_length
                        preds_list.append('O')
                        # print('Exception', ex, beginning_index, out_label_preds)
                    words_list.append(word)
                return words_list, preds_list

            words_list, preds_list = realign(text_sentence, out_label_preds)
            predicted_entities = list(zip(words_list, preds_list))

            # print(predicted_entities)

            for entity in predicted_entities:
                if entity[1] != 'O':
                    entity_json = {
                        "entity": entity[0],
                        "prediction": entity[1],
                        "start_pos": sentence.find(entity[0]),
                        "end_pos": sentence.find(entity[0]) + len(entity[0]),
                        "id": ci["id"] + ":2637:2642:newsag:bert"
                    }
                    print(entity_json)
                    result_json["entities"].append(entity_json)

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