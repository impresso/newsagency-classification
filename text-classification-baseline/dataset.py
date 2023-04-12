import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import (LabelEncoder,
                                   MultiLabelBinarizer)

COLUMNS = ["TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "NE-FINE-LIT",
                   "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED",
                   "NEL-LIT", "NEL-METO", "MISC"]
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

        indexes = list(range(len(COLUMNS)))

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

    def __init__(self, train_dataset, dev_dataset, test_dataset, tokenizer, max_len, mode='train'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        columns = ["TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "NE-FINE-LIT",
                   "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED",
                   "NEL-LIT", "NEL-METO", "MISC"]
        indexes = list(range(len(columns)))

        self.train_phrases = _read_conll(
            train_dataset,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)
        self.dev_phrases = _read_conll(
            dev_dataset,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)
        self.test_phrases = _read_conll(
            test_dataset,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)

        self.train_sequence_targets = [int(item[-1]) for item in self.train_phrases]
        self.train_token_targets = [item[1][1] for item in self.train_phrases]
        self.train_tokens = [item[1][0] for item in self.train_phrases]

        self.dev_sequence_targets = [int(item[-1]) for item in self.dev_phrases]
        self.dev_token_targets = [item[1][1] for item in self.dev_phrases]
        self.dev_tokens = [item[1][0] for item in self.dev_phrases]

        self.test_sequence_targets = [int(item[-1]) for item in self.test_phrases]
        self.test_token_targets = [item[1][1] for item in self.test_phrases]
        self.test_tokens = [item[1][0] for item in self.test_phrases]

        unique_token_labels = set(sum(self.train_token_targets, []))
        self.label_map = dict(
            zip(unique_token_labels, range(len(unique_token_labels))))

        self.train_token_targets = [[self.label_map[element]
                               for element in item[1][1]] for item in self.train_phrases]
        self.dev_token_targets = [[self.label_map[element]
                               for element in item[1][1]] for item in self.dev_phrases]
        self.test_token_targets = [[self.label_map[element]
                               for element in item[1][1]] for item in self.test_phrases]

    def __len__(self):
        return len(self.train_phrases)

    def get_label_map(self):
        return self.label_map

    def get_inverse_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def change_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            sequence = self.train_tokens[index]
            # if not self.test:
            sequence_targets = self.train_sequence_targets[index]
            token_targets = self.train_token_targets[index]
        elif self.mode == 'dev':
            sequence = self.dev_tokens[index]
            # if not self.test:
            sequence_targets = self.dev_sequence_targets[index]
            token_targets = self.dev_token_targets[index]
        else:
            sequence = self.test_tokens
            # if not self.test:
            sequence_targets = self.test_sequence_targets[index]
            token_targets = self.test_token_targets[index]

        encoding = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,  # TODO: add token type ids
            truncation=True
        )

        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(
                    encoding['attention_mask'],
                    dtype=torch.long)
        offset_mapping = torch.tensor(encoding['offset_mapping'], dtype=torch.long)
        offset_mapping = torch.sub(torch.transpose(offset_mapping, 0, 1)[1],
                                   torch.transpose(offset_mapping, 0, 1)[0])

        if self.mode in ['test']:
            return {
                'sequence': ' '.join(sequence),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'offset_mapping': offset_mapping}
        else:
            # print(token_targets)
            # Pad the tensor with zeros until the maximum length
            token_targets = torch.tensor(token_targets, dtype=torch.long)
            # print(input_ids.shape, token_targets.shape)
            padded_token_targets = torch.zeros(self.max_len, dtype=torch.long)
            padded_token_targets[:token_targets[:self.max_len].size(
                0)] = token_targets[:self.max_len]

            sequence_targets = torch.tensor(sequence_targets, dtype=torch.long)

            assert input_ids.shape == attention_mask.shape
            # assert sequence_targets.shape == attention_mask.shape
            assert padded_token_targets.shape == input_ids.shape

            return {
                'sequence': ' '.join(sequence),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'sequence_targets': sequence_targets,
                'token_targets': padded_token_targets,
                'offset_mapping': offset_mapping}

    def get_info(self):
        num_sequence_labels = len(set(self.train_sequence_targets))
        num_token_labels = len(set(sum(self.train_token_targets, [])))
        return num_sequence_labels, num_token_labels

    def get_dataframe(self):
        return self.df
