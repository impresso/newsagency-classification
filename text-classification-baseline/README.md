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
      --evaluate_during_training
```

`--model_name_or_path`: the preffered languge model (by default, `bert-base-cased`);\
Other models can be found at [HuggingFace](https://huggingface.co/), such as models trained on [historical documents](https://huggingface.co/dbmdz/). To change a model, one needs to specify the name in the HuggingFace site, e.g., for [hmBERT](https://huggingface.co/dbmdz/bert-base-historic-multilingual-cased), `--model_name_or_path dbmdz/bert-base-historic-multilingual-cased`;\
`--train_dataset`, `--train_dataset`, and `--train_dataset`: point to the path of the *.tsv files;\
`--output_dir`: points to the folder where the experiments (models and predictions) are saved;\
`--device`: can be `cuda` or `cpu`

## Evaluation with HIPE-scorer:
```
python clef_evaluation.py --ref ../data/newsagency/newsagency-data-2-dev-fr.tsv \
      --pred ../experiments/model_bert_base_cased_max_sequence_length_64_epochs_3/newsagency-data-2-dev-fr_pred.tsv \
      --task nerc_coarse --outdir ../experiments/model_bert_base_cased_max_sequence_length_64_epochs_3 \
      --hipe_edition HIPE-2022 --log ../experiments/model_bert_base_cased_max_sequence_length_64_epochs_3/logs_scorer.txt
```
For fine-grained, change to --task nerc_fine
