#!/bin/bash

# Define an array of max_seq_len values
max_seq_len_values=(128 256 512 1024)

# Define a range of logging_suffix values
logging_suffix_values=(1 2 3 4 5)

# Define a list of models
models=("dbmdz/bert-base-french-europeana-cased" "bert-base-cased")

# Get the language from the first command line argument
language=$1

# Check if language parameter was provided
if [ -z "$language" ]
then
    echo "No language parameter provided. Please run the script with a language parameter. e.g., ./run.sh fr"
    exit 1
fi

# Loop over the models array
for model in "${models[@]}"
do
    # Replace '/' in the model name with '-'
    model_path="${model//\//-}"

    # Loop over the max_seq_len_values array
    for max_seq_len in "${max_seq_len_values[@]}"
    do
        # Loop over the logging_suffix_values array
        for logging_suffix in "${logging_suffix_values[@]}"
        do
            echo "Running experiment with model = $model, max_seq_len = $max_seq_len, language = $language and logging_suffix = $logging_suffix"

            CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python3 main.py \
                --model_name_or_path $model \
                --train_dataset data/newsagency/newsagency-data-2-test-$language-copy.tsv \
                --dev_dataset data/newsagency/newsagency-data-2-dev-$language.tsv \
                --test_dataset data/newsagency/newsagency-data-2-test-$language.tsv \
                --output_dir experiments \
                --device cuda \
                --train_batch_size 16 \
                --logging_steps 10 \
                --max_seq_len $max_seq_len \
                --logging_suffix $logging_suffix \
                --evaluate_during_training \
                --do_train

            echo "Running evaluation for model = $model, max_seq_len = $max_seq_len, language = $language and logging_suffix = $logging_suffix"

            python3 HIPE-scorer/clef_evaluation.py \
                --ref ../data/newsagency/newsagency-data-2-dev-$language.tsv \
                --pred ../experiments/${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-2-dev-$language_pred.tsv \
                --task nerc_coarse \
                --outdir ../experiments/${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ../experiments/${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_dev_scorer.txt

            python3 HIPE-scorer/clef_evaluation.py \
                --ref ../data/newsagency/newsagency-data-2-test-$language.tsv \
                --pred ../experiments/${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-2-test-$language_pred.tsv \
                --task nerc_coarse \
                --outdir ../experiments/${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ../experiments/${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_test_scorer.txt
        done
    done
done
