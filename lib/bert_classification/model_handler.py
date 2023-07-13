import json
import torch
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import os
import sys
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
import json
import os
import string
import re
import torch.nn.functional as F
# Get the directory of your script
import os
import sys
# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)
# add the current directory to sys.path
sys.path.insert(0, current_directory)
# from cli_tagger_local import get_entities, realign, tokenize

def tokenize(text):
    print(text)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ' + punctuation + ' ')
    return text.split()


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


def realign(text_sentence, out_label_preds, TOKENIZERS, language, reverted_label_map):
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

class NewsAgencyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.tokenizers = {}

    def initialize(self, ctx):
        # Load models for French and German
        # self.models['fr'] = torch.jit.load(ctx.manifest['models']['fr'], map_location=self.device)
        # self.models['de'] = torch.jit.load(ctx.manifest['models']['de'], map_location=self.device)
        # boilerplate
        properties = ctx.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id")) if torch.cuda.is_available() else self.map_location
        )
        self.manifest = ctx.manifest
        # model_dir is the inside of your archive!
        # extra-files are in this dir.
        model_dir = properties.get("model_dir")

        serialized_file = self.manifest["model"]["serializedFile"]

        # Load corresponding tokenizers
        MODEL_PATHS = {
            'fr': 'agency-fr',
            'de': 'agency-de'}

        self.tokenizers['fr'] = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizers['de'] = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        self.models['fr'] = torch.jit.load(serialized_file, map_location=self.device)
        self.models['de'] = torch.jit.load(serialized_file, map_location=self.device)


        self.models['fr'].to(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        self.models['de'].to(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        self.models['fr'].eval()
        self.models['de'].eval()

        self.label_map = {"B-org.ent.pressagency.Reuters": 0, "B-org.ent.pressagency.Stefani": 1, "O": 2,
                 "B-org.ent.pressagency.Extel": 3, "B-org.ent.pressagency.Havas": 4, "I-org.ent.pressagency.Xinhua": 5, "I-org.ent.pressagency.Domei": 6, "B-org.ent.pressagency.Belga": 7, "B-org.ent.pressagency.CTK": 8, "B-org.ent.pressagency.ANSA": 9, "B-org.ent.pressagency.DNB": 10, "B-org.ent.pressagency.Domei": 11, "I-pers.ind.articleauthor": 12, "I-org.ent.pressagency.Wolff": 13, "B-org.ent.pressagency.unk": 14, "I-org.ent.pressagency.Stefani": 15, "I-org.ent.pressagency.AFP": 16, "B-org.ent.pressagency.UP-UPI": 17, "I-org.ent.pressagency.ATS-SDA": 18, "I-org.ent.pressagency.unk": 19, "B-org.ent.pressagency.DPA": 20, "B-org.ent.pressagency.AFP": 21, "I-org.ent.pressagency.DNB": 22, "B-pers.ind.articleauthor": 23, "I-org.ent.pressagency.UP-UPI": 24, "B-org.ent.pressagency.Kipa": 25, "B-org.ent.pressagency.Wolff": 26, "B-org.ent.pressagency.ag": 27, "I-org.ent.pressagency.Extel": 28, "I-org.ent.pressagency.ag": 29, "B-org.ent.pressagency.ATS-SDA": 30, "I-org.ent.pressagency.Havas": 31, "I-org.ent.pressagency.Reuters": 32, "B-org.ent.pressagency.Xinhua": 33, "B-org.ent.pressagency.AP": 34, "B-org.ent.pressagency.APA": 35, "I-org.ent.pressagency.ANSA": 36, "B-org.ent.pressagency.DDP-DAPD": 37, "I-org.ent.pressagency.TASS": 38, "I-org.ent.pressagency.AP": 39, "B-org.ent.pressagency.TASS": 40, "B-org.ent.pressagency.Europapress": 41, "B-org.ent.pressagency.SPK-SMP": 42}

        self.reverted_label_map = {v: k for k, v in dict(self.label_map).items()}


    def preprocess(self, requests):

        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            # text = data[0]['body']
            print(data)
            print('-----', input_text, type(input_text))
            input_text = json.loads(input_text)
            text = input_text['body']
            text_sentence = tokenize(text)

            language = input_text['language']  # Assuming that the request contains the 'language' field
            tokenized_inputs = self.tokenizers[language](
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                # We use this argument because the texts in our dataset are lists
                # of words (with a label for each word).
                # is_split_into_words=True
            )
            input_ids = torch.tensor([tokenized_inputs['input_ids']]).to(
                'cuda' if torch.cuda.is_available() else 'cpu')

            print(input_ids)
            return input_ids, text_sentence, language

    def inference(self, inputs):
        # Choose model based on 'language' field in inputs
        print(inputs)
        input_ids, text_sentence, language = inputs

        with torch.no_grad():
            outputs = self.models[language](input_ids)

        sequence_result, tokens_result = outputs[0], outputs[1]

        return sequence_result, tokens_result, text_sentence, language

    def postprocess(self, outputs):
        # postprocess the outputs here, for example, convert predictions to labels
        # outputs = ... # some processing here

        sequence_result, tokens_result, text_sentence, language = outputs
        token_logits = tokens_result['logits']
        out_token_preds = token_logits.detach().cpu().numpy()
        out_label_preds = np.argmax(out_token_preds, axis=2)[0]

        # apply softmax to convert logits to probabilities
        # probabilities = F.softmax(token_logits, dim=-1)
        # confidence_scores = probabilities.max(dim=-1)[0].cpu().numpy()

        # print(confidence_scores, len(confidence_scores[0]), len(out_label_preds))

        words_list, preds_list = realign(text_sentence, out_label_preds,
                                         self.tokenizers, language, self.reverted_label_map)

        entities = get_entities(words_list, preds_list)

        return entities



