import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import (LabelEncoder,
                                   MultiLabelBinarizer)


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
        # if self.test_set:
        #     df = pd.read_csv(dataset, sep="\t", names=["id", "text"])
        #     self.classes = None
        #     self.encoded_classes = None
        # else:
        #     df = pd.read_csv(dataset, sep="\t", names=["id", "text", "labels"])
        #
        #     self.label_encoder = MultiLabelBinarizer()
        #     df['labels'] = self.label_encoder.fit_transform(df['labels'])
        #     self.classes = self.label_encoder.classes_
        #     self.encoded_classes = pd.unique(df['labels'])
        #     self.targets = df['label'].to_numpy()
        #
        # self.sequences = df['text'].to_numpy()
        # self.df = df
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
        # take the last element which says if the sentence contains entities or not

        self.token_targets = [item[1][1] for item in self.phrases]
        self.tokens = [item[1][0] for item in self.phrases]

        # import pdb;pdb.set_trace()

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, index):
        sequence = str(self.sequences[index])
        if not self.test:
            sequence_targets = self.sequence_targets[index]
            token_targets = self.token_targets[index]

        encoding = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        # import pdb;pdb.set_trace()
        if self.test_set:
            return {
                'sequence': sequence,
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
            }
        else:
            return {
                'sequence': sequence,
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'sequence_targets': torch.tensor(sequence_targets, dtype=torch.long),
                'token_targets': torch.tensor(token_targets, dtype=torch.long)
            }

    def get_info(self):
        num_sequence_labels = len(set(self.sequence_targets))
        num_token_labels = len(set(sum(self.token_targets, [])))
        return num_sequence_labels, num_token_labels

    def get_dataframe(self):
        return self.df
