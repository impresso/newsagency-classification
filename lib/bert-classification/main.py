from utils import set_seed, SEED
from model import train, evaluate
import argparse
import torch
from model import ModelForSequenceAndTokenClassification
import os
from dataset import NewsDataset
import logging
from transformers import AutoTokenizer, AutoConfig, AdamW
import yaml
import json
from utils import write_predictions


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
    parser.add_argument('--label_map',
                        type=str,
                        default='data/label_map.json',
                        help="Path to the *json file for the label mapping.")
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
        '--checkpoint',
        type=str,
        default='',
        help="The folder with a checkpoint model to be loaded and continue training or evaluate.")
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
        "--logging_suffix",
        type=str,
        default="",
        help="Suffix to further specify name of the folder where the logging is stored."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.")
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
    
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--continue_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to make experiment reproducible.")
    

    args = parser.parse_args()

    set_seed(args.seed)

    args.model_name_or_path = args.model_name_or_path.lower()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(
        args.output_dir,
        "model_{}_max_sequence_length_{}_epochs_{}_run{}".format(
            args.model_name_or_path.replace('/', '_').replace('-', '_'),
            args.max_sequence_len,
            args.epochs,
            args.logging_suffix),
            )
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    

    #logging.basicConfig(level=logging.INFO)
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(os.path.join(args.output_dir, "logging.log")),
        logging.StreamHandler()
    ]
    )
    logger = logging.getLogger(__name__)

    logging.info(
        "Trained models and results will be saved in {}.".format(
            args.output_dir))
    
    # Setup CUDA, GPU & distributed training
    if args.device == 'cpu':
        device = torch.device("cpu")
        args.n_gpu = 1
    elif args.local_rank == -1 or args.device == 'cuda':
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

    #if label map was not specified, generate it from data and save it in output folder
    if not args.label_map:
        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len)
    
        label_map = train_dataset.get_label_map()

        # dataset, tokenizer, max_len, test = False, label_map = None
        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map)
        
        label_map = dev_dataset.get_label_map()

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map)

        label_map = test_dataset.get_label_map()
    #if specified, load the label map and use it for the data
    else:
        label_map = json.load(open(args.label_map, "r"))

        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map)

        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map)


        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map)

    json.dump(label_map, open(os.path.join(args.output_dir, 'data/label_map.json'), "w"))

    num_sequence_labels, num_token_labels = test_dataset.get_info()

    logging.info(
        "Number of unique token labels {}, number of unique sequence labels {}.".format(
            num_token_labels,
            num_sequence_labels))

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        problem_type="single_label_classification")

    model = ModelForSequenceAndTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        num_sequence_labels=num_sequence_labels,
        num_token_labels=num_token_labels)

    model = model.to(args.device)

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

    if args.do_train:
        train(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            label_map=label_map)

    elif args.continue_train:
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        train(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            label_map=label_map)
    else:
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        # checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        config = AutoConfig.from_pretrained(
            args.checkpoint,
            problem_type="single_label_classification",
            local_files_only=True)

        model = ModelForSequenceAndTokenClassification.from_pretrained(
            args.checkpoint,
            config=config,
            num_sequence_labels=num_sequence_labels,
            num_token_labels=num_token_labels,
            local_files_only=True)

        model = model.to(args.device)

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)

        #dev data
        results, words_list, preds_list, report_bin, report_class = evaluate(args, model, dev_dataset, label_map, tokenizer=tokenizer)

        write_predictions(args.output_dir, dev_dataset.get_filename(), words_list, preds_list)

        results_devset = dict()
        results_devset["global"] = results
        results_devset["sent-level"] = report_bin
        results_devset["token-level"] = report_class


        #test data
        results, words_list, preds_list, report_bin, report_class = evaluate(
        args, model, test_dataset, label_map, tokenizer=tokenizer)

        write_predictions(args.output_dir, test_dataset.get_filename(), words_list, preds_list)

        results_testset = dict()
        results_testset["global"] = results
        results_testset["sent-level"] = report_bin
        results_testset["token-level"] = report_class


        #results to json
        all_results = {"dev": results_devset, "test": results_testset}
        if "-de" in test_dataset.get_filename():
            with open(os.path.join(args.output_dir, "all_results_de.json"), "w") as f:
                json.dump(all_results, f)
        elif "-fr" in test_dataset.get_filename():
            with open(os.path.join(args.output_dir, "all_results_fr.json"), "w") as f:
                json.dump(all_results, f)
        else:
            logger.info(f"Was not able to deduct language from filename of testset, thus no metrics were saved.")


