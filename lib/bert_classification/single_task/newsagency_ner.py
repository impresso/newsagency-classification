from transformers import Pipeline
import numpy as np
import torch
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import string
import torch.nn.functional as F

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


def get_entities(tokens, tags, confidences):
    """postprocess the outputs here, for example, convert predictions to labels
    [
        {
            "entity": "B-org.ent.pressagency.AFP",
            "score": 0.99669313,
            "index": 13,
            "word": "AF",
            "start": 43,
            "end": 45,
        },
        {
            "entity": "I-org.ent.pressagency.AFP",
            "score": 0.42747754,
            "index": 14,
            "word": "##P",
            "start": 45,
            "end": 46,
        },
    ]

    [[('AFP', 'org.ent.pressagency.AFP', (12, 13), (47, 50))]]
    """
    tags = [tag.replace("S-", "B-").replace("E-", "I-") for tag in tags]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    entities = []
    idx: int = 0
    char_position = 0  # This will hold the current character position

    for subtree in ne_tree:
        # skipping 'O' tags
        if isinstance(subtree, Tree):
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])

            entity_start_position = char_position
            entity_end_position = entity_start_position + len(original_string)

            entities.append(
                {
                    "entity": original_label,
                    "score": np.average(confidences[idx : idx + len(subtree)]),
                    "index": idx,
                    "word": original_string,
                    "start": entity_start_position,
                    "end": entity_end_position,
                }
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


def realign(
    text_sentence, out_label_preds, softmax_scores, tokenizer, reverted_label_map
):
    preds_list, words_list, confidence_list = [], [], []
    word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
    for idx, word in enumerate(text_sentence):

        try:
            beginning_index = word_ids.index(idx)
            preds_list.append(reverted_label_map[out_label_preds[beginning_index]])
            confidence_list.append(softmax_scores[0][beginning_index].max())
        except Exception as ex:  # the sentence was longer then max_length
            preds_list.append("O")
            confidence_list.append(0.0)
        words_list.append(word)
    return words_list, preds_list, confidence_list


class NewsAgencyModelPipeline(Pipeline):

    def _sanitize_parameters(self, **kwargs):
        # Add any additional parameter handling if necessary
        return kwargs, {}, {}

    def preprocess(self, text, **kwargs):
        tokenized_inputs = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=256
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
        """
        Postprocess the outputs of the model
        :param outputs:
        :param kwargs:
        :return:
        """
        tokens_result, text_sentence = outputs

        # Get raw logits and convert to numpy array
        logits = tokens_result["logits"].detach().cpu().numpy()

        # Compute the most likely token ids
        tokens_result = np.argmax(logits, axis=2)[0]

        # Calculate softmax scores for better interpretability
        softmax_scores = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

        words_list, preds_list, confidence_list = realign(
            text_sentence,
            tokens_result,
            softmax_scores,
            self.tokenizer,
            id2label,
        )

        entities = get_entities(words_list, preds_list, confidence_list)

        return entities
