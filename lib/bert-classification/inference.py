# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoConfig
from model import ModelForSequenceAndTokenClassification
import json
import os

import torch
from transformers import AutoTokenizer, AutoConfig
from model import ModelForSequenceAndTokenClassification
import json
import os
import pysbd


class SingletonModel:
    _instance = None
    _model = None

    @staticmethod
    def getInstance(model_path='bert-base-uncased', label_map_path='data/label_map.json'):
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
                    num_sequence_labels=len(json.load(open(label_map_path, "r"))),
                    num_token_labels=len(json.load(open(label_map_path, "r"))),
                    local_files_only=True)

                SingletonModel._model = SingletonModel._model.to('cuda' if torch.cuda.is_available() else 'cpu')
                SingletonModel._model.eval()

                traced_model = torch.jit.trace(SingletonModel._model, [torch.zeros((1, 1), dtype=torch.long)])
                torch.jit.save(traced_model, traced_model_path)
                SingletonModel._model = torch.jit.load(traced_model_path)

            SingletonModel._instance = self


singleton_model = SingletonModel.getInstance()


def predict_entities(article_id, article_text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    with open('data/label_map.json', "r") as f:
        label_map = json.load(f)

    seg = pysbd.Segmenter(language="en", clean=False)
    sentences = seg.segment(article_text)

    result_json = {"article_id": article_id, "entities": []}

    for sentence in sentences:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])

        with torch.no_grad():
            outputs = singleton_model._model([input_ids])
        logits = outputs.logits
        predicted_indices = torch.argmax(logits, dim=1)

        predicted_labels = [label_map[str(index)] for index in predicted_indices]

        words = tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_entities = list(zip(words, predicted_labels))

        for entity in predicted_entities:
            entity_json = {
                "entity": entity[0],
                "prediction": entity[1],
                "start_pos": sentence.find(entity[0]),
                "end_pos": sentence.find(entity[0]) + len(entity[0])
            }
            result_json["entities"].append(entity_json)

    return result_json
