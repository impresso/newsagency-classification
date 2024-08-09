## Models

For training the models, first copy the scripts from (scripts)[https://github.com/impresso/newsagency-classification/tree/main/lib/bert_classification/scripts] in this folder and run both `run.sh` and `run_multilingual.sh`. 

```
bash run.sh de
bash run.sh fr
bash run_multilingual.sh
```

Both scripts will create a new folder called `experiments` where the models will be saved under subfolders with a specific name that represents the msin hyperparameters (eg `experiments/model_bert_base_german_cased_max_sequence_length_256_epochs_3_run887`). Each experiment subfolder will contain different checkpoints. The best model is usually the last saved checkpoint.

## HuggingFace

To push to HF, run `python push_model_to_hf.py`. For now, the script pushes the best models --the latest checkpoints-- (ie trained with multilingual DeBERTa) and will push them in `impresso-project` on HF.

### Example run:
```python
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false python main.py \
                --model_name_or_path dbmdz/bert-medium-historic-multilingual-cased \
                --train_dataset ../../../data/annotated_data/multilingual/newsagency-data-train-multilingual.tsv \
                --dev_dataset ../../../data/annotated_data/de/newsagency-data-dev-de.tsv \
                --test_dataset ../../../data/annotated_data/fr/newsagency-data-test-fr.tsv \
                --label_map ../../../data/annotated_data/label_map.json \
                --output_dir experiments \
                --device cuda \
                --train_batch_size 32 \
                --logging_steps 536 \
                --save_steps 536 \
                --max_sequence_len 512 \
                --logging_suffix multilingual_run \
                --evaluate_during_training  \
                --seed 42 \
                --do_train

```
