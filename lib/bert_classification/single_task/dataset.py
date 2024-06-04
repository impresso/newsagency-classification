import torch
from torch.utils.data import Dataset

COLUMNS = ["TOKEN",
           "NE-COARSE-LIT",
           "NE-COARSE-METO",
           "NE-FINE-LIT",
           "NE-FINE-METO",
           "NE-FINE-COMP",
           "NE-NESTED",
           "NEL-LIT",
           "NEL-METO",
           "RENDER",
           "SEG",
           "OCR-INFO",
           "MISC"]


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

        # print(sample)
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
                        "# id",
                        "# "]):
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
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


class NewsDataset(Dataset):
    """
    """

    def __init__(self, tsv_dataset, tokenizer,
                 max_len,
                 test=False,
                 label_map={}):
        """
        Initiliazes a dataset in IOB format.
        :param tsv_dataset: tsv filename of the train/test/dev dataset
        :param tokenizer: the LM tokenizer
        :param max_len: the maximum sequence length, get be 512 for BERT-based LMs
        :param test: if it is the test dataset or not - can be disconsidered for now
        :param label_map: the label map {0: 'B-pers'}
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test = test

        indexes = list(range(len(COLUMNS)))

        self.tsv_dataset = tsv_dataset

        self.phrases = _read_conll(
            tsv_dataset,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)

        self.sequence_targets = [int(item[-1]) for item in self.phrases]
        self.token_targets = [item[1][3] for item in self.phrases]
        self.tokens = [item[1][0] for item in self.phrases]

        self.label_map = label_map
        unique_token_labels = set(sum(self.token_targets, []))
        label_mapped = dict(
            zip(unique_token_labels, range(len(unique_token_labels))))
        missed_labels = set(label_mapped) - set(label_map)

        print("Appended following labels to label_map:", missed_labels)

        num_labels = len(self.label_map)
        for i, missed_label in enumerate(missed_labels):
            self.label_map[missed_label] = num_labels + i

        self.token_targets = [[self.label_map[element]
                               for element in item[1][3]] for item in self.phrases]

    def __len__(self):
        return len(self.phrases)

    def get_filename(self):
        return self.tsv_dataset

    def get_label_map(self):
        return self.label_map

    def get_inverse_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def tokenize_and_align_labels(self, sequence, token_targets):
        """
        :param sequence:
        :param token_targets:
        :return:
        """
        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word).
            is_split_into_words=True,
            return_token_type_ids=True
        )
        labels = []
        label_all_tokens = False
        # for i, label in enumerate(tokens):
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                labels.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                labels.append(token_targets[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    labels.append(token_targets[word_idx])
                else:
                    labels.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs["token_targets"] = labels
        return tokenized_inputs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        sequence = self.tokens[index]
        sequence_targets = self.sequence_targets[index]
        token_targets = self.token_targets[index]

        encoding = self.tokenize_and_align_labels(sequence, token_targets)

        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        token_type_ids = torch.tensor(
            encoding['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(
            encoding['attention_mask'], dtype=torch.long)

        token_targets = torch.tensor(
            encoding['token_targets'], dtype=torch.long)

        sequence_targets = torch.tensor(sequence_targets, dtype=torch.long)

        assert input_ids.shape == attention_mask.shape
        assert token_targets.shape == input_ids.shape

        return {
            'sequence': ' '.join(sequence),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sequence_targets': sequence_targets,
            'token_targets': token_targets,
            'token_type_ids': token_type_ids}

    def get_info(self):
        """
        :return:
        """
        num_sequence_labels = len(set(self.sequence_targets))
        num_token_labels = len(self.label_map)
        return num_sequence_labels, num_token_labels
