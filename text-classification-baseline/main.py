import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import os
from dataset import NewsDataset
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.INFO)
def run(
        model,
        data_loader,
        optimizer,
        device,
        scheduler,
        n_examples,
        mode='train'):
    print("Training the Model")
    if mode == 'train':
        model = model.train()
    else:
        model = model.eval()
    losses = []
    correct_predictions = 0
    for data in tqdm(data_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["target"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        # the second return value is logits
        _, preds = torch.max(outputs[1], dim=1)
        loss = outputs[0]  # the first return value is loss
        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--train', type=str, default='')
    parser.add_argument('--dev', type=str, default='')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--max_sequence_len', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    parser.add_argument('--result', type=str, default='')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_warmup_steps', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.model = args.model.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_set = NewsDataset(args.train, tokenizer, args.max_sequence_len)
    classes, encoded_classes, train_set_shape = train_set.get_info()

    logging.info("Label Encoding of {} --> {}".format(classes, str(np.sort(encoded_classes))))

    encoded_classes = encoded_classes.astype(str)
    logging.info("Shape of the train set: {}".format(train_set_shape))
    train_data_loader = DataLoader(train_set, args.train_batch_size, shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(encoded_classes))
    model = model.to(args.device)
