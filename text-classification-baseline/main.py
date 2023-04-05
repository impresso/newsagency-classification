import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model import ModelForSequenceAndTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import os
from dataset import NewsDataset
from torch.utils.data import DataLoader
import uuid
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help="The model to be loaded. It can be a pre-trained model "
        "or a fine-tuned model (folder on the disk).")
    parser.add_argument('--train_dataset',
                        type=str,
                        default='',
                        help="Path to the *csv or *tsv train file.")
    parser.add_argument('--dev_dataset',
                        type=str,
                        default='',
                        help="Path to the *csv or *tsv dev file.")
    parser.add_argument('--test_dataset',
                        type=str,
                        default='',
                        help="Path to the *csv or *tsv test file.")
    parser.add_argument('--max_sequence_len',
                        type=int, default=64,
                        help="Maximum text length.")
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help="Number of epochs. Default to 3 (can be 5 - max 10)")
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.")
    parser.add_argument(
        '--valid_batch_size',
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help="The folder where the experiment details and the predictions should be saved.")
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help="The learning rate - it can go from 2e-5 to 3e-5.")
    parser.add_argument(
        '--n_warmup_steps',
        type=int,
        default=0,
        help="The warmup steps - the number of steps in on epoch or 0.")
    parser.add_argument(
        '--device',
        default='cuda',
        help="The device on which should the model run - cpu or cuda.")

    args = parser.parse_args()
    args.model = args.model.lower()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(
        args.output_dir,
        "model_{}_max_sequence_length_{}_epochs_{}_uuid_{}".format(
            args.model.replace(
                '/',
                '_').replace(
                '-',
                '_'),
            args.max_sequence_len,
            args.epochs,
            str(
                uuid.uuid4())))
    os.mkdir(args.output_dir)

    logging.info(
        "Trained models adn results will be saved in {}.".format(
            args.output_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_set = NewsDataset(args.train_dataset, tokenizer, args.max_sequence_len)
    num_sequence_labels, num_token_labels = train_set.get_info()

    # logging.info("Label Encoding of {} --> {}".format(classes,
    #              str(np.sort(encoded_classes))))
    #
    # encoded_classes = encoded_classes.astype(str)
    # logging.info("Shape of the train set: {}".format(train_set_shape))
    train_data_loader = DataLoader(
        train_set,
        args.train_batch_size,
        shuffle=False,
        num_workers=0)

    model = ModelForSequenceAndTokenClassification.from_pretrained(args.model,
                                                                   problem_type="single_label_classification",
                                                                   num_sequence_labels=num_sequence_labels,
                                                                   num_token_labels=num_token_labels)
    #, num_labels=len(encoded_classes)
    model = model.to(args.device)

    import pdb
    pdb.set_trace()
