from transformers import (
    AutoTokenizer,
    Pipeline,
)
import numpy as np
import torch
from torch import nn
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import string

label2id = {
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
    "B-org.ent.pressagency.SPK-SMP": 42,
}

id2label = {v: k for k, v in label2id.items()}


def tokenize(text):
    # print(text)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    return text.split()


def get_entities(tokens, tags):
    tags = [tag.replace("S-", "B-").replace("E-", "I-") for tag in tags]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    entities = []
    idx = 0
    char_position = 0  # This will hold the current character position

    for subtree in ne_tree:
        # skipping 'O' tags
        if isinstance(subtree, Tree):
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])

            entity_start_position = char_position
            entity_end_position = entity_start_position + len(original_string)

            entities.append(
                (
                    original_string,
                    original_label,
                    (idx, idx + len(subtree)),
                    (entity_start_position, entity_end_position),
                )
            )
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


def realign(text_sentence, out_label_preds, tokenizer, reverted_label_map):
    preds_list, words_list, confidence_list = [], [], []
    word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
    for idx, word in enumerate(text_sentence):

        try:
            beginning_index = word_ids.index(idx)
            preds_list.append(reverted_label_map[out_label_preds[beginning_index]])
        except Exception as ex:  # the sentence was longer then max_length
            preds_list.append("O")
        words_list.append(word)
    return words_list, preds_list


class NewsAgencyModelPipeline(Pipeline):
    # def __init__(self, model_id, config, **kwargs):
    #     super().__init__(model_id, config, **kwargs)
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    #
    #     self.model = ModelForSequenceAndTokenClassification.from_pretrained(
    #         model_id,
    #         num_sequence_labels=2,
    #         num_token_labels=len(label2id),
    #     )
    #     self.model.eval()  # Set the model to evaluation mode
    # def __init__(self, model, tokenizer, **kwargs):
    #     super().__init__(self, model, tokenizer, **kwargs)
    #     self.model = model
    #     self.tokenizer = tokenizer

    def _sanitize_parameters(self, **kwargs):
        # Add any additional parameter handling if necessary
        return kwargs, {}, {}

    def preprocess(self, text, **kwargs):
        tokenized_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word).
            # is_split_into_words=True
        )

        text_sentence = tokenize(text)
        return tokenized_inputs, text_sentence

    def _forward(self, inputs):
        inputs, text_sentence = inputs
        input_ids = torch.tensor([inputs["input_ids"]], dtype=torch.long).to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs, text_sentence

    def postprocess(self, outputs, **kwargs):
        # postprocess the outputs here, for example, convert predictions to labels
        # outputs = ... # some processing here

        outputs, text_sentence = outputs
        try:
            _, tokens_result = outputs[0], outputs[1]
        except:
            tokens_result = outputs[0]

        tokens_result = np.argmax(
            tokens_result["logits"].detach().cpu().numpy(), axis=2
        )[0]

        words_list, preds_list = realign(
            text_sentence,
            tokens_result,
            self.tokenizer,
            id2label,
        )

        entities = get_entities(words_list, preds_list)
        # print('*'*20, 'Result:', entities)

        return [entities]

    # def postprocess(self, outputs, **kwargs):
    #
    #     # Extract and process logits
    #     outputs, inputs = outputs[0], outputs[1]
    #
    #     token_logits, sequence_logits = outputs[0], outputs[1]
    #
    #     token_logits = token_logits.logits.detach().cpu().numpy()
    #     sequence_logits = sequence_logits.logits.detach().cpu().numpy()
    #
    #     text_sentences = [
    #         self.tokenizer.convert_ids_to_tokens(input_ids)
    #         for input_ids in inputs["input_ids"].detach().cpu().numpy()
    #     ]
    #
    #     sequence_preds = np.argmax(token_logits, axis=-1)
    #     token_preds = np.argmax(sequence_logits, axis=1)
    #
    #     # sequence_preds = torch.argmax(sequence_logits, dim=-1)
    #     # token_preds = torch.argmax(token_logits, dim=-1)
    #
    #     preds_list = [[] for _ in range(token_preds.shape[0])]
    #     words_list = [[] for _ in range(token_preds.shape[0])]
    #
    #     for idx_sentence, item in enumerate(zip(text_sentences, token_preds)):
    #         text_sentence, out_label_preds = item
    #         word_ids = self.tokenizer(
    #             text_sentence, is_split_into_words=True
    #         ).word_ids()
    #         for idx, word in enumerate(text_sentence):
    #             beginning_index = word_ids.index(idx)
    #
    #             try:
    #                 preds_list[idx_sentence].append(
    #                     id2label[out_label_preds[beginning_index]]
    #                 )
    #             except BaseException:  # the sentence was longer then max_length
    #                 preds_list[idx_sentence].append("O")
    #             words_list[idx_sentence].append(word)
    #
    #     import pdb
    #
    #     pdb.set_trace()
    #     return {
    #         "sequence_classification": sequence_preds.cpu().numpy(),
    #         "token_classification": token_preds.cpu().numpy(),
    #     }
