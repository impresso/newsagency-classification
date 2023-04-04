## Classification and Exploration of News Agency Content

This repository holds the code related to the master project of Lea Marxen on the classification and exploration of news agency content, based on the _impresso_ corpus.

## Organization

`inception`: ..
`notebooks`: ..
`samples`: ..


### Data Specification

### Data Annotation

### Text Classification Baseline

The model is based on `AutoModelForSequenceClassification` provided by the `transformers` library and it is a generic model class that will be instantiated as one of the sequence classification model classes of the library when created with the `from_pretrained(pretrained_model_name_or_path)` method.

The classification baseline has three modes:
- binary classification: is there any news agency mentioned in the text of not
- multiclass classification: a text can only belong to a news agency
- multicalss and multilabel classification: a text can belong to several news agencies
