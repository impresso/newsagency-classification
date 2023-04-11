import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import (LabelEncoder,
                                   MultiLabelBinarizer)
from transformers.utils import PaddingStrategy

def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):

        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]

        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:

        sample = []
        start = next(f).strip()  # Skip columns
        start = next(f).strip()

        data = []
        for line_idx, line in enumerate(f, 0):
            line = line.strip()

            if any(
                    substring in line for substring in [
                        'DOCSTART',
                        '###',
                        "# id",
                        "# ",
                        '###']):
                continue

            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                has_entities = all(v == 'O' for v in res[1])
                                data.append([line_idx, res, has_entities])
                                # import pdb;pdb.set_trace()
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            elif 'EndOfSentence' in line:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                has_entities = all(v == 'O' for v in res[1])
                                data.append([line_idx, res, has_entities])
                                # import pdb;
                                # pdb.set_trace()
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        has_entities = all(v == 'O' for v in res[1])
                        data.append([line_idx, res, has_entities])
                        # import pdb;
                        # pdb.set_trace()
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


class NERDataset(Dataset):

    def __init__(self, filename):
        columns = ["TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "NE-FINE-LIT",
                   "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED",
                   "NEL-LIT", "NEL-METO", "MISC"]
        indexes = list(range(len(columns)))

        self.phrases = _read_conll(
            filename,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, index):
        phrase = str(self.phrases[index])

        return phrase

    def get_info(self):
        return self.phrases

    def get_dataframe(self):
        return self.phrases


class NewsDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, test=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test = test
        columns = ["TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "NE-FINE-LIT",
                   "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED",
                   "NEL-LIT", "NEL-METO", "MISC"]
        indexes = list(range(len(columns)))

        self.phrases = _read_conll(
            dataset,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)

        self.sequence_targets = [int(item[-1]) for item in self.phrases]
        # take the last element which says if the sentence contains entities or
        # not

        self.token_targets = [item[1][1] for item in self.phrases]
        self.tokens = [item[1][0] for item in self.phrases]

        unique_token_labels = set(sum(self.token_targets, []))
        self.label_map = dict(
            zip(unique_token_labels, range(len(unique_token_labels))))

        self.token_targets = [[self.label_map[element]
                               for element in item[1][1]] for item in self.phrases]

    def __len__(self):
        return len(self.phrases)

    def get_label_map(self):
        return self.label_map

    def get_inverse_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def tokenize(self, words):
        """ tokenize input"""

        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, words):
        """ preprocess """
        tokens, valid_positions = self.tokenize(words)

        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)

        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_len:
            input_ids.append(0)
            attention_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return {'sequence': words,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'valid_positions': valid_positions}


    def __getitem__(self, index):
        sequence = self.tokens[index]

        if not self.test:
            sequence_targets = self.sequence_targets[index]
            token_targets = self.token_targets[index]

        # encoding = self.preprocess(sequence)

        encoding = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=PaddingStrategy.MAX_LENGTH,
            return_offsets_mapping=True,
            return_token_type_ids=True,  # TODO: add token type ids
            is_split_into_words=True,
            truncation=True,
            # return_tensors="pt"
        )

        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        # print(input_ids.shape)
        padded_input_ids = torch.zeros(self.max_len, dtype=torch.long)
        padded_input_ids[:input_ids[:self.max_len].size(0)] = input_ids[:self.max_len]

        offset_mapping = torch.tensor(encoding['offset_mapping'], dtype=torch.long)
        offset_mapping = torch.sub(torch.transpose(offset_mapping, 0, 1)[1],
                                   torch.transpose(offset_mapping, 0, 1)[0])

        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        padded_attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        padded_attention_mask[:attention_mask[:self.max_len].size(0)] = attention_mask[:self.max_len]

        if self.test:
            return {
                'sequence': sequence,
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask}
        else:
            # Pad the tensor with zeros until the maximum length

            token_targets = torch.tensor(token_targets, dtype=torch.long)
            padded_token_targets = torch.zeros(self.max_len, dtype=torch.long)
            padded_token_targets[:token_targets[:self.max_len].size(0)] = token_targets[:self.max_len]

            sequence_targets = torch.tensor(sequence_targets, dtype=torch.long)
            padded_sequence_targets = torch.zeros(self.max_len, dtype=torch.long)
            padded_sequence_targets[0] = sequence_targets

            # import pdb;pdb.set_trace()
            print(padded_token_targets.shape, type(padded_token_targets))
            print(attention_mask.shape, type(attention_mask))
            print(offset_mapping.shape, type(offset_mapping))
            print(sequence_targets.shape, sequence_targets, type(sequence_targets))
            print('-'*10)

            return {
                'sequence': sequence,
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask,
                'sequence_targets': sequence_targets,
                'token_targets': padded_token_targets}

    def get_info(self):
        num_sequence_labels = len(set(self.sequence_targets))
        num_token_labels = len(set(sum(self.token_targets, [])))
        return num_sequence_labels, num_token_labels

    def get_dataframe(self):
        return self.df
