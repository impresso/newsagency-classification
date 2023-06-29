## Classification and Exploration of News Agency Content

This repository holds the code related to the master project of Lea Marxen on the classification and exploration of news agency content, based on the _impresso_ corpus.

## Organization

- `annotation_settings/`: Contains the planning and settings for annotation with Inception.
  - `inception_settings/`: Contains specifications for inception settings, e.g. the TypeSystem and tagset.
  - `Annotation Guidelines for Newsagency Classification v2.pdf`
  - `annotation_planning_per_annotator.csv`
  - `annotation_planning_per_doc.csv`
- `data/`: Contains part of the data used during the project.
  - `annotation/`
  - `sampling/`
  - `split/`
- `lib/`: Contains python scripts for the classification, conversion of data and helpers.
  - `bert-classification/`
  - `helpers/`
  - `inception_postprocessing/`
  - `inception_preprocessing/`
- `notebooks/`: Contains the notebooks used in the creation of the training corpus, annotation and analysis.
  - `1_sampling_training_data/`
  - `2_annotation/`
  - `3_classification/`
  - `4_analysis/`



### Data Specification

### Data Annotation

### Text Classification Baseline

The model is based on `AutoModelForSequenceClassification` provided by the `transformers` library and it is a generic model class that will be instantiated as one of the sequence classification model classes of the library when created with the `from_pretrained(pretrained_model_name_or_path)` method.

The classification baseline has three modes:
- binary classification: is there any news agency mentioned in the text of not
- multiclass classification: a text can only belong to a news agency
- multicalss and multilabel classification: a text can belong to several news agencies
