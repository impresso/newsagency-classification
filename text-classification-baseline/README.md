## Multilabel, multiclass, and multitask classification
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


### Training a model
```
TOKENIZERS_PARALLELISM=false python main.py \
      --train_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
      --dev_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv \
      --test_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
      --output_dir experiments \
      --device cpu \
      --train_batch_size 16 \
      --logging_steps 100 \
      --evaluate_during_training
```

`--train_dataset`, `--train_dataset`, and `--train_dataset`: point to the path of the *.tsv files;\
`--output_dir`: points to the folder where the experiments (models and predictions) are saved;\
`--device`: can be `cuda` or `cpu`
