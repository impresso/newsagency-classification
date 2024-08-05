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
