#!/bin/bash
#
# Define an array of max_seq_len values
max_seq_len_values=(64 128 256 512)
#
# Define a range of logging_suffix values
logging_suffix_values=(1 2 3 4 5)
#
# Get the language from the first command line argument
language=$1
gpu=$2
#
# Check if language parameter was provided
if [ -z "$language" ]
then
    echo "No language parameter provided. Please run the script with a language parameter. e.g., ./run.sh fr"
    exit 1
fi

if [ $language == "fr" ]
then
    # Define a list of models
    #models=("xlm-roberta-base")
    #models=("dbmdz/bert-base-french-europeana-cased")
    models=("bert-base-multilingual-cased" "dbmdz/bert-base-french-europeana-cased" "bert-base-cased" "camembert-base" "dbmdz/bert-base-historic-multilingual-cased" "xlm-roberta-base")
    log_steps=1477 #1465*4
fi

if [ $language == "de" ]
then
    models=("dbmdz/bert-base-german-europeana-cased" "bert-base-cased" "bert-base-german-cased" "bert-base-multilingual-cased" "dbmdz/bert-base-historic-multilingual-cased" "xlm-roberta-base")
    #models=("bert-base-german-cased") # "xlm-roberta-base")
    log_steps=666 #584*4
fi

#
# Loop over the models array
for model in "${models[@]}"
do    
    # Replace '/' and '-' in the model name with '_'
    model_path="${model//[\/-]/_}"

    # Loop over the max_seq_len_values array
    for max_seq_len in "${max_seq_len_values[@]}"
    do  
        <<Block_comment # all with batch size 2
        #smaller batch size for max_seq_len 512
        if [ $max_seq_len == 512 ]
        then
            batch_size=8
        else
            batch_size=16
        fi
Block_comment

        # Loop over the logging_suffix_values array
        for run in "${logging_suffix_values[@]}"
        do
            logging_suffix=_${language}_$run

            echo "Running experiment with model = $model, max_seq_len = $max_seq_len, language = $language and logging_suffix = $logging_suffix"

            CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false python3 main.py \
                --model_name_or_path $model \
                --train_dataset ../../data/annotated_data/$language/newsagency-data-train-$language.tsv \
                --dev_dataset ../../data/annotated_data/$language/newsagency-data-dev-$language.tsv \
                --test_dataset ../../data/annotated_data/$language/newsagency-data-test-$language.tsv \
                --label_map ../../data/annotated_data/label_map.json \
                --output_dir experiments \
                --device cuda \
                --train_batch_size 16 \
                --logging_steps $log_steps \
                --save_steps $log_steps \
                --max_sequence_len $max_seq_len \
                --logging_suffix $logging_suffix \
                --evaluate_during_training  \
                --seed $run \
                --do_train

            echo "Running evaluation for model = $model, max_seq_len = $max_seq_len, language = $language and logging_suffix = $logging_suffix"

            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/$language/newsagency-data-dev-$language.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-dev-${language}_pred.tsv \
                --task nerc_fine \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_dev_scorer.txt

            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/$language/newsagency-data-test-$language.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-test-${language}_pred.tsv \
                --task nerc_fine \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_test_scorer.txt
        done
    done
done
