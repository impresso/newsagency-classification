# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoConfig
from model import ModelForSequenceAndTokenClassification
import json
import os


def predict_entities(article_id, article_text, model_path='bert-base-uncased', label_map_path='data/label_map.json'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    input_ids = tokenizer.encode(article_text, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])

    traced_model_path = "traced_model.pt"

    # Try loading the model as Torchscript first
    if os.path.exists(traced_model_path):
        try:
            model = torch.jit.load(traced_model_path)
        except Exception:
            model = None
    else:
        model = None

    # If loading as Torchscript failed, load with from_pretrained
    if model is None:
        config = AutoConfig.from_pretrained(
            model_path,
            problem_type="single_label_classification",
            local_files_only=True)

        model = ModelForSequenceAndTokenClassification.from_pretrained(
            model_path,
            config=config,
            num_sequence_labels=len(label_map),
            num_token_labels=len(label_map),
            local_files_only=True)

        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Creating the trace
        model.eval()
        traced_model = torch.jit.trace(model, [input_ids])

        # Save Torchscript model
        torch.jit.save(traced_model, traced_model_path)

        # Load Torchscript model
        model = torch.jit.load(traced_model_path)

    with torch.no_grad():
        outputs = model([input_ids])
    logits = outputs.logits
    predicted_indices = torch.argmax(logits, dim=1)

    # Convert indices to labels
    predicted_labels = [label_map[str(index)] for index in predicted_indices]

    # Combine words with their predictions
    words = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_entities = list(zip(words, predicted_labels))

    # Constructing JSON output
    result_json = {"article_id": article_id, "entities": []}

    for entity in predicted_entities:
        entity_json = {
            "entity": entity[0],
            "prediction": entity[1],
            "start_pos": article_text.find(entity[0]),
            "end_pos": article_text.find(entity[0]) + len(entity[0])
        }
        result_json["entities"].append(entity_json)

    return result_json
