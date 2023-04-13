import argparse
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          AutoConfig,
                          AdamW,
                          get_linear_schedule_with_warmup)
from model import ModelForSequenceAndTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from seqeval.metrics import classification_report as seq_classification_report
from torch import nn
from collections import defaultdict
from tqdm import tqdm, trange
from datetime import datetime
import os
from dataset import NewsDataset
import uuid
import logging
import math
from torch.utils.data.distributed import DistributedSampler
import random
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset)
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from dataset import COLUMNS

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from seqeval.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SEED = 42


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed(42)


def write_predictions(dataset, words_list, preds_list):
    with open(args.dev_dataset, 'r') as f:
        tsv_lines = f.readlines()

    flat_words_list = [item for sublist in words_list for item in sublist]
    flat_preds_list = [item for sublist in preds_list for item in sublist]
    with open(args.dev_dataset.replace('.tsv', '_pred.tsv'), 'w') as f:
        idx = 0
        for idx_tsv_line, tsv_line in enumerate(tsv_lines):
            if idx_tsv_line == 0:
                f.write(tsv_line)
            elif len(tsv_line.split('\t')) != len(COLUMNS):
                f.write(tsv_line)
            elif len(tsv_line.strip()) == 0:
                f.write(tsv_line)
            else:
                try:
                    f.write(
                        flat_words_list[idx] +
                        '\t' +
                        flat_preds_list[idx] +
                        '\n')
                except BaseException:
                    import pdb
                    pdb.set_trace()
                idx += 1
                f.flush()


def evaluate(
        args,
        model,
        dataset,
        label_map,
        mode='dev',
        prefix="",
        tokenizer=None):
    # eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(
        dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    out_sequence_ids, out_token_ids = None, None
    out_sequence_preds, out_token_preds = None, None
    sentences, text_sentences = None, None
    offset_mappings = None
    model.eval()

    # finish = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(
                    args.device), "attention_mask": batch["attention_mask"].to(
                    args.device), "sequence_labels": batch["sequence_targets"].to(
                    args.device), "token_labels": batch["token_targets"].to(
                    args.device), 'token_type_ids': batch['token_type_ids'].to(device)}

            # import
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet"] else None
            #     )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            # tmp_eval_loss, logits = outputs[:2]

            sequence_result, tokens_result = outputs[0], outputs[1]
            token_logits = tokens_result.logits

            # the second return value is logits
            sequence_logits = sequence_result.logits

            # correct_predictions += torch.sum(sequence_preds == sequence_targets)

            tmp_eval_loss = sequence_result.loss

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        if out_token_preds is None:
            out_token_preds = token_logits.detach().cpu().numpy()
            out_sequence_preds = sequence_logits.detach().cpu().numpy()

            out_token_ids = inputs["token_labels"].detach().cpu().numpy()
            out_sequence_ids = inputs["sequence_labels"].detach().cpu().numpy()

            sentences = [tokenizer.convert_ids_to_tokens(
                input_ids) for input_ids in inputs["input_ids"].detach().cpu().numpy()]
            text_sentences = [text.split(' ') for text in batch["sequence"]]

            # offset_mappings = inputs["offset_mapping"].detach().cpu().numpy()
        else:
            out_token_preds = np.append(
                out_token_preds, token_logits.detach().cpu().numpy(), axis=0)
            out_sequence_ids = np.append(
                out_sequence_ids,
                inputs["sequence_labels"].detach().cpu().numpy(),
                axis=0)

            out_token_ids = np.append(
                out_token_ids,
                inputs["token_labels"].detach().cpu().numpy(),
                axis=0)
            out_sequence_preds = np.append(
                out_sequence_preds,
                sequence_logits.detach().cpu().numpy(),
                axis=0)

            sentences = np.append(sentences, [tokenizer.convert_ids_to_tokens(
                input_ids) for input_ids in inputs["input_ids"].detach().cpu().numpy()], axis=0)
            text_sentences = np.append(
                text_sentences, [
                    text.split(' ') for text in batch["sequence"]], axis=0)

            # offset_mappings = np.append(
            #     offset_mappings,
            #     inputs["offset_mapping"].detach().cpu().numpy(),
            #     axis=0)
        # finish += 1

        # if finish == 20:
        #     break
        #

    out_token_preds = np.argmax(out_token_preds, axis=2)
    out_sequence_preds = np.argmax(out_sequence_preds, axis=1)

    logger.info('Evaluation for yes/no classification.')
    report = classification_report(
        out_sequence_ids, out_sequence_preds, digits=4)
    logger.info("\n%s", report)

    eval_loss = eval_loss / nb_eval_steps

    label_map = {label: i for i, label in label_map.items()}

    out_label_list = [[] for _ in range(out_token_ids.shape[0])]
    preds_list = [[] for _ in range(out_token_ids.shape[0])]
    words_list = [[] for _ in range(out_token_ids.shape[0])]

    for idx_sentence, item in enumerate(
            zip(text_sentences, out_token_ids, out_token_preds)):
        text_sentence, out_label_ids, out_label_preds = item
        word_ids = tokenizer(
            text_sentence,
            is_split_into_words=True).word_ids()
        for idx, word in enumerate(text_sentence):
            beginning_index = word_ids.index(idx)
            try:
                out_label_list[idx_sentence].append(
                    label_map[out_label_ids[beginning_index]])
            except BaseException:  # the sentence was longer then max_length
                out_label_list[idx_sentence].append('O')
            try:
                preds_list[idx_sentence].append(
                    label_map[out_label_preds[beginning_index]])
            except BaseException:  # the sentence was longer then max_length
                preds_list[idx_sentence].append('O')
            words_list[idx_sentence].append(word)

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info('Evaluation for named entity recognition & classification.')
    report = seq_classification_report(out_label_list, preds_list, digits=4)
    logger.info("\n%s", report)

    logger.info('Evaluation for named entity recognition classification.')
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, words_list, preds_list


def train(
        args,
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        tokenizer,
        labels):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # data_collator = DataCollatorForTokenClassification(
    #     tokenizer, pad_to_multiple_of=(8 if args.fp16 else None)
    # )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)

    t_total = math.ceil(len(train_dataset) / \
                        args.train_batch_size) * args.epochs # assume 10 epochs
    # 10% of training steps for warmup
    num_warmup_steps = math.ceil(t_total * 0.1)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], "weight_decay": args.weight_decay, }, {
            "params": [
                p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0}, ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(
            args.model_name_or_path,
            "optimizer.pt")) and os.path.isfile(
            os.path.join(
                args.model_name_or_path,
                "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    args.model_name_or_path,
                    "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(
                os.path.join(
                    args.model_name_or_path,
                    "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[1]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info(
        "  Gradient Accumulation steps = %d",
        args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model
        # path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(
        args.epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # import pdb;pdb.set_trace()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = {"input_ids": batch["input_ids"].to(
                args.device), "attention_mask": batch["attention_mask"].to(
                args.device), "sequence_labels": batch["sequence_targets"].to(
                args.device), "token_labels": batch["token_targets"].to(
                args.device), 'token_type_ids': batch['token_type_ids'].to(device)}

            outputs = model(**inputs)

            sequence_result, tokens_result = outputs[0], outputs[1]
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = sequence_result.loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [
                    -1,
                        0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not
                    # average well
                    if (args.local_rank == -
                            1 and args.evaluate_during_training):

                        results, words_list, preds_list = evaluate(
                            args, model, dev_dataset, labels, mode="dev", tokenizer=tokenizer)

                        write_predictions(dev_dataset, words_list, preds_list)

                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step)

                    tb_writer.add_scalar(
                        "lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1,
                                       0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(
                        args, os.path.join(
                            output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(
                            output_dir, "optimizer.pt"))
                    torch.save(
                        scheduler.state_dict(), os.path.join(
                            output_dir, "scheduler.pt"))
                    logger.info(
                        "Saving optimizer and scheduler states to %s",
                        output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    results, words_list, preds_list = evaluate(
        args, model, test_dataset, labels, mode="test", tokenizer=tokenizer)

    write_predictions(test_dataset, words_list, preds_list)

    return global_step, tr_loss / global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
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
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
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
        '--eval_batch_size',
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help="The folder where the experiment details and the predictions should be saved.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--max_steps",
        default=-
        1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        '--n_warmup_steps',
        type=int,
        default=0,
        help="The warmup steps - the number of steps in on epoch or 0.")
    parser.add_argument("--local_rank", type=int, default=-
                        1, help="For distributed training: local_rank")

    parser.add_argument(
        '--device',
        default='cuda',
        help="The device on which should the model run - cpu or cuda.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )

    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path.lower()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(
        args.output_dir,
        "model_{}_max_sequence_length_{}_epochs_{}".format(
            args.model_name_or_path.replace(
                '/',
                '_').replace(
                '-',
                '_'),
            args.max_sequence_len,
            args.epochs))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logging.info(
        "Trained models and results will be saved in {}.".format(
            args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.device == 'cuda':
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = NewsDataset(
        dataset=args.train_dataset,
        tokenizer=tokenizer,
        max_len=args.max_sequence_len)

    num_sequence_labels, num_token_labels = train_dataset.get_info()

    label_map = train_dataset.get_label_map()
    # dataset, tokenizer, max_len, test = False, label_map = None
    dev_dataset = NewsDataset(
        dataset=args.dev_dataset,
        tokenizer=tokenizer,
        max_len=args.max_sequence_len,
        label_map=label_map)

    test_dataset = NewsDataset(
        dataset=args.test_dataset,
        tokenizer=tokenizer,
        max_len=args.max_sequence_len,
        label_map=label_map)

    logging.info(
        "Number of unique token labels {}, number of unique sequence labels {}.".format(
            num_token_labels,
            num_sequence_labels))

    # train_data_loader = DataLoader(
    #     train_dataset,
    #     args.train_batch_size,
    #     shuffle=True,
    #     num_workers=os.cpu_count())

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        problem_type="single_label_classification")

    model = ModelForSequenceAndTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        num_sequence_labels=num_sequence_labels,
        num_token_labels=num_token_labels)

    model = model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], "weight_decay": args.weight_decay, }, {
            "params": [
                p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0}, ]

    train(
        args,
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        tokenizer,
        label_map)
