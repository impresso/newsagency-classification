import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import (LabelEncoder,
                                   MultiLabelBinarizer)


class NewsDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, test_set=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test_set = test_set
        if self.test_set:
            df = pd.read_csv(dataset, sep="\t", names=["id", "text"])
            self.classes = None
            self.encoded_classes = None
        else:
            df = pd.read_csv(dataset, sep="\t", names=["id", "text", "labels"])

            self.label_encoder = MultiLabelBinarizer()
            df['labels'] = self.label_encoder.fit_transform(df['labels'])
            self.classes = self.label_encoder.classes_
            self.encoded_classes = pd.unique(df['labels'])
            self.targets = df['label'].to_numpy()

        self.sequences = df['text'].to_numpy()
        self.df = df

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = str(self.sequences[index])
        if not self.test_set:
            target = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
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
                'target': torch.tensor(target, dtype=torch.long)
            }

    def get_info(self):
        return self.classes, self.encoded_classes, self.df.shape

    def get_dataframe(self):
        return self.df
