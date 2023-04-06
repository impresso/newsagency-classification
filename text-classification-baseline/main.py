import argparse
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          AutoConfig,
                          AdamW,
                          get_linear_schedule_with_warmup)
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
import math
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

def eval(model):
    pass

def run(model,
        data_loader,
        optimizer,
        device,
        scheduler,
        n_examples,
        mode='train'):
    print("Training the Model")
    if mode == 'train':
        model.train()
    else:
        model.eval()
    losses = []
    correct_predictions = 0
    # for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    for data in tqdm(data_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        sequence_targets = data["sequence_targets"].to(device)
        token_targets = data["token_targets"].to(device)

        # import pdb;pdb.set_trace()

        # padded_tensor = torch.zeros((token_targets.size(0), 64))
        # padded_tensor[:, :token_targets[:, :64].size(1)] = token_targets[:, :64]
        # padded_tensor = torch.zeros(64)
        # padded_tensor[:token_targets[:64].size(0)] = token_targets[:64]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_labels=sequence_targets,
            token_labels=token_targets
        )
        sequence_result, tokens_result = outputs[0], outputs[1]
        sequence_logits  = sequence_result.logits
        token_logits  = tokens_result.logits

        # the second return value is logits
        _, sequence_preds = torch.max(sequence_logits, dim=1).detach().cpu().numpy()
        correct_predictions += torch.sum(sequence_preds == sequence_targets)

        token_logits = torch.argmax(F.log_softmax(token_logits, dim=2), dim=2)
        token_logits = token_logits.detach().cpu().numpy()
        import pdb;pdb.set_trace()


        if mode == 'train':
            loss = sequence_result.loss  # the first return value is loss
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            pass

    if mode == 'train':
        return correct_predictions.double() / n_examples, np.mean(losses)
    else:
        return correct_predictions.double() / n_examples



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

    train_dataset = NewsDataset(args.train_dataset, tokenizer, args.max_sequence_len)
    num_sequence_labels, num_token_labels = train_dataset.get_info()

    logging.info("Number of unique token labels {}, number of unique sequence labels {}.".format(num_token_labels,
                                                                                                 num_sequence_labels))

    train_data_loader = DataLoader(
        train_dataset,
        args.train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count())

    config = AutoConfig.from_pretrained('bert-base-uncased', problem_type="single_label_classification")

    model = ModelForSequenceAndTokenClassification.from_pretrained(args.model,
                                                                   config = config,
                                                                   num_sequence_labels=num_sequence_labels,
                                                                   num_token_labels=num_token_labels)

    model = model.to(args.device)
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    num_training_steps = math.ceil(len(train_dataset) / args.train_batch_size) * args.epochs  # assume 10 epochs
    num_warmup_steps = math.ceil(num_training_steps * 0.1)  # 10% of training steps for warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)


    run(model,
        train_data_loader,
        optimizer,
        args.device,
        scheduler,
        len(train_dataset),
        mode='train')