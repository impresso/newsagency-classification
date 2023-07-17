from model import ModelForSequenceAndTokenClassification
from transformers import AutoTokenizer, AutoConfig
import torch
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


sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000

# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)
# add the current directory to sys.path
sys.path.insert(0, current_directory)

model_paths = {
    'fr': 'agency-fr/',
    'de': 'agency-de/'}
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

_tokenizers = {}
_models = {}

for language in ['fr', 'de']:
    # Load the tokenizer
    _tokenizers[language] = AutoTokenizer.from_pretrained(
        model_paths[language], local_files_only=True)

    traced_model_path = f"agency-{language}/traced_model_{language}.pt"

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
            num_token_labels=len(label_map),
            local_files_only=True)

        _models[language] = _models[language]

        scripted_model = torch.jit.trace(
            _models[language], [
                torch.zeros(
                    (1, 1), dtype=torch.long)], strict=False)
        torch.jit.save(scripted_model, traced_model_path)
