#!/bin/bash
#
# Define an array of max_seq_len values
max_seq_len_values=(64 128 256 512)
#
# Define a range of logging_suffix values
logging_suffix_values=(1 2 3 4 5)
#
# Get the language from the first command line argument
language="multilingual"
log_steps=8204 #2051*4
#models=("bert-base-multilingual-cased")
models=("bert-base-cased" "bert-base-multilingual-cased" "dbmdz/bert-base-historic-multilingual-cased" "xlm-roberta-base")

#
# Loop over the models array
for model in "${models[@]}"
do    
    # Replace '/' and '-' in the model name with '_'
    model_path="${model//[\/-]/_}"

    # Loop over the max_seq_len_values array
    for max_seq_len in "${max_seq_len_values[@]}"
    do  
        <<Block_comment #batch size will be set to 2
        #smaller batch size for max_seq_len 512
        if [ $max_seq_len == 512 ]
        then
            batch_size=8
            checkpoint="checkpoint-12306"
        else
            batch_size=16
            checkpoint="checkpoint-6153"
        fi
Block_comment
        checkpoint="checkpoint-6429"
        # Loop over the logging_suffix_values array
        for run in "${logging_suffix_values[@]}"
        do

            logging_suffix=_${language}_$run
            echo "Running experiment with model = $model, max_seq_len = $max_seq_len, language = $language and logging_suffix = $logging_suffix"

            #if only want to test on fr dataset, comment this block
            #<<Block_comment 
            CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python main.py \
                --model_name_or_path $model \
                --train_dataset ../../data/annotated_data/newsagency-data-train-all-fr.tsv \
                --dev_dataset ../../data/annotated_data/de/newsagency-data-dev-de.tsv \
                --test_dataset ../../data/annotated_data/de/newsagency-data-test-de.tsv \
                --label_map ../../data/annotated_data/label_map.json \
                --output_dir experiments \
                --device cuda \
                --train_batch_size 16 \
                --logging_steps 2143 \
                --save_steps 2143 \
                --max_sequence_len $max_seq_len \
                --logging_suffix $logging_suffix \
                --evaluate_during_training \
                --seed $run \
                --do_train --do_eval

            echo "Running evaluation for model = $model, max_seq_len = $max_seq_len, language = $language and logging_suffix = $logging_suffix"

            #evaluation for French dataset
            CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python main.py \
                --model_name_or_path $model \
                --train_dataset ../../data/annotated_data/newsagency-data-train-all-fr.tsv \
                --dev_dataset ../../data/annotated_data/fr/newsagency-data-dev-fr.tsv \
                --test_dataset ../../data/annotated_data/fr/newsagency-data-test-fr.tsv \
                --label_map ../../data/annotated_data/label_map.json \
                --output_dir experiments \
                --device cuda \
                --train_batch_size 16 \
                --logging_steps 2143 \
                --save_steps 2143 \
                --max_sequence_len $max_seq_len \
                --logging_suffix $logging_suffix \
                --evaluate_during_training \
                --checkpoint ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/$checkpoint \
                --seed $run \
                --do_eval

            #evaluation on German dataset
            # nerc-fine
            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/de/newsagency-data-dev-de.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-dev-de_pred.tsv \
                --task nerc_fine \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_dev_scorer.txt

            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/de/newsagency-data-test-de.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-test-de_pred.tsv \
                --task nerc_fine \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_test_scorer.txt

            #evaluation on French dataset
            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/fr/newsagency-data-dev-fr.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-dev-fr_pred.tsv \
                --task nerc_fine \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_dev_scorer.txt

            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/fr/newsagency-data-test-fr.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-test-fr_pred.tsv \
                --task nerc_fine \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_test_scorer.txt

            #evaluation on German dataset
            # nerc-coarse
            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/de/newsagency-data-dev-de.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-dev-de_pred.tsv \
                --task nerc_coarse \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_dev_scorer.txt

            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/de/newsagency-data-test-de.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-test-de_pred.tsv \
                --task nerc_coarse \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_test_scorer.txt

            #evaluation on French dataset
            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/fr/newsagency-data-dev-fr.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-dev-fr_pred.tsv \
                --task nerc_coarse \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_dev_scorer.txt

            python HIPE-scorer/clef_evaluation.py \
                --ref ../../data/annotated_data/fr/newsagency-data-test-fr.tsv \
                --pred ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/newsagency-data-test-fr_pred.tsv \
                --task nerc_coarse \
                --outdir ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix} \
                --hipe_edition HIPE-2022 \
                --log ./experiments/model_${model_path}_max_sequence_length_${max_seq_len}_epochs_3_run${logging_suffix}/logs_test_scorer.txt


        done
    done
done
