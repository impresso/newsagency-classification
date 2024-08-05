## Multitask classification (binary and multiclass, sequence-based and token-based classification)
#### of news agencies content at token-level and article-level

## hipe2020 working example
- the data is annotated at token-level in the [CoNLL format](https://universaldependencies.org/format.html) with [IOB tagging format](https://www.geeksforgeeks.org/nlp-iob-tags/) for several types of entities (pers, org, loc, etc.).
- since there are no article-level annotations, we consider the presence of specific types of entities as the definition of a class.

#### Example:

```csv
FAITS	O	O	O	O	O	O	_	_	_
DIVERS	O	O	O	O	O	O	_	_	EndOfLine
La	O	O	O	O	O	O	_	_	_
panique	O	O	O	O	O	O	_	_	_
des	O	O	O	O	O	O	_	_	_
éléphants	O	O	O	O	O	O	_	_	_
au	O	O	O	O	O	O	_	_	_
grand	O	O	O	O	O	O	_	_	EndOfLine
cortège	O	O	O	O	O	O	_	_	_
dc	O	O	O	O	O	O	_	_	_
Munich	B-loc	O	B-loc.adm.town	O	O	O	Q1726	_	NoSpaceAfter
.	O	O	O	O	O	O	_	_	EndOfSentence
```
This sentence has one coarse entity of type `loc`, thus the article-level class is `has_locations`.

### Language model-based classifier

The multiclass classifier will have two predictions heads:
- for predicting at token-level the entities (Munich is a `loc`)
- for predicting at article-level the class (`has_locations`)

## How-tos

First, you need an environement with `python >= 3.7` on the server, and activate it.
`source activate YOUR_ENV`

Next, if you do not have already the project on `/scratch/USERNAME`, you can move them from your computer with `scp`. Example:
`scp -r newsagency-classification/text-classification-baseline/ USERNAME@iccluster040.iccluster.epfl.ch:/scratch/USERNAME/OTHER_FOLDERS/newsagency-classification/`

Install the requirements:
`pip install -r requirements.txt`

Training a model based on `bert-base-cased`:
```
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python main.py \
      --model_name_or_path bert-base-cased\
      --train_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
      --dev_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv \
      --test_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
      --output_dir experiments \
      --device cuda \
      --train_batch_size 16 \
      --logging_steps 100 \
      --evaluate_during_training \
      --do_train
```
`CUDA_VISIBLE_DEVICES=1`: choose on which GPU will the model be trained, if `CUDA_VISIBLE_DEVICES` is not mentioned, the model will be trained in parallel on all available GPUs.
`--model_name_or_path`: the languge model (by default, `bert-base-cased`);\
Other models can be found at [HuggingFace](https://huggingface.co/), such as models trained on [historical documents](https://huggingface.co/dbmdz/). To change a model, one needs to specify the name in the HuggingFace site, e.g., for [hmBERT](https://huggingface.co/dbmdz/bert-base-historic-multilingual-cased), `--model_name_or_path dbmdz/bert-base-historic-multilingual-cased`;\
`--train_dataset`, `--dev_dataset`, and `--test_dataset`: point to the path of the *.tsv files;\
`--output_dir`: points to the folder where the experiments (models and predictions) are saved;\
`--device`: can be `cuda` or `cpu`
`--do_classif`: for performing sentence classification (a sentence can contain or not a mention of a news agency

## Evaluation with HIPE-scorer:
```
python clef_evaluation.py \
      --ref ../data/newsagency/newsagency-data-dev-fr.tsv \
      --pred ../experiments/model_bert_base_cased_max_sequence_length_64_epochs_3/newsagency-data-dev-fr_pred.tsv \
      --task nerc_coarse \
      --outdir ../experiments/model_bert_base_cased_max_sequence_length_64_epochs_3 \
      --hipe_edition HIPE-2022 \
      --log ../experiments/model_bert_base_cased_max_sequence_length_64_epochs_3/logs_scorer.txt
```
For fine-grained, change to --task nerc_fine

## Running with `bash` scripts 
Both to train and evaluate the models

```
bash run.sh de
bash run.sh fr
bash run_multilingual.sh
```

`run_multilingual.sh` will train on a dataset with French and German articles and evaluate on French and German separately.

## Inference (TorchServe + Dask)

Files:
```
export_model.py
cli_tagger.py
model_handler.py
```

Necessary libs:

java
```
sudo apt-get update
sudo apt-get install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
```

python >= 3.10
```
pip install nvgpu
pip install pynvml==11.0.0
pip install onnxruntime
pip install dask
python -m pip install "dask[distributed]" —upgrade
pip install torchserve torch-model-archiver
```

TorchServe is a flexible and easy-to-use tool for serving and scaling PyTorch models in production. The current models `newsagency-model-fr` and `newsagency-model-de` are in the associated folders with the same name. The naming convention that TorchServe accepts is similar to HuggingFace models, therefore, the folders containing the models can only be folders with the same names as the models (`newsagency-model-fr` and `newsagency-model-de`) and they need to be in the same folder where TorchServe is started. TorchServe accepts *pth models but it prefers scripted models with TorchScript. TorchScript is a way to create serializable and optimizable models from PyTorch code.

The models were converted with TorchScript and `export_models.py` saves the checkpoint files in torchscript (`traced_model_de.pt`, `traced_model_fr.pt`) in the same folders.
```
python export_models.py
```

The models need a handler, code that will let TorchServe know what to do when a model receives an API call: `model_handler.py`

Start:
```
torchserve --start --ncs --model-store model_store  --models agency_fr=newsagency-model-fr.mar agency_de=newsagency-model-de.mar 
```
TorchServe loads `newsagency-model-fr.mar` and `newsagency-model-de.mar`, each with the same model handler.

Stop:
```
torchserve --stop
```

API calls depend on the language of the text:
```
http://127.0.0.1:8080/predictions/agency_{language}'

{'text': TEXT, 'language': LANGUAGE}
```

A key feature of TorchServe is the ability to package all model artifacts into a single model archive file. It is a separate command line interface (CLI), torch-model-archiver, that can take model checkpoints or model definition file with state_dict, and package them into a .mar file. This file can then be redistributed and served by anyone using TorchServe. It takes in the following model artifacts: a model checkpoint file in case of torchscript or a model definition file and a state_dict file in case of eager mode, and other optional assets that may be required to serve the model. The CLI creates a .mar file that TorchServe's server CLI uses to serve the models.

```
torch-model-archiver --model-name newsagency-model-fr --version 1.0 --serialized-file newsagency-model-fr/traced_model_fr.pt --handler model_handler --force --extra-files "newsagency-model-fr/traced_model_fr.pt,newsagency-model-fr/tokenizer_config.json,newsagency-model-fr/tokenizer.json,newsagency-model-fr/vocab.txt,newsagency-model-fr/special_tokens_map.json,newsagency-model-fr/config.json,newsagency-model-fr/traced_model_fr.pt" --export-path model_store

torch-model-archiver --model-name newsagency-model-de --version 1.0 --serialized-file newsagency-model-de/traced_model_de.pt --handler model_handler --force --extra-files "newsagency-model-de/traced_model_de.pt,newsagency-model-de/tokenizer_config.json,newsagency-model-de/tokenizer.json,newsagency-model-de/vocab.txt,newsagency-model-de/special_tokens_map.json,newsagency-model-de/config.json,newsagency-model-de/traced_model_de.pt" --export-path model_store
```
Then, different processes were started for ˜2200 files, in batches of 100.

```
python cli_tagger.py --input_dir=DATA_FOLDER --output_dir=na_mentions/ --logfile=log-test.log --workers 64 --prefix 03
```
where `DATA_FOLDER` contains the `.json` archives that pack around 16,000 articles each, the number of workers is the number of CPUs and the prefix takes all files starting with `03` (100 files).

We ran 8 `cli_tagger.py` in parallel, each handling 100 files with `dask` scheduler. The generation of the predictions took 4 days.
