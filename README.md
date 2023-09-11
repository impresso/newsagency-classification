## Classification and Exploration of News Agency Content

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python)](https://www.python.org/) 
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.3-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.13/) 
[![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

This repository holds the code related to the master project of Lea Marxen on the classification and exploration of news agency content, based on the _impresso_ corpus. 
The project was implemented in the summer semester 2023, with the supervision of Maud Ehrmann, Emanuela Boros and Marten Düring.

## About

Since their beginnings in the 1830s and 1840s, news agencies have played an important role in the national and international news market, aiming to deliver news as fast and as reliable as possible. While we know that newspapers have been using agency content for a long time to produce their stories, the amount to which the agencies shape our news often remains unclear. Although researchers have already addressed this question, recently by using computational methods to assess the influence of news agencies at present, large-scale studies on the role of news agencies in the past continue to be rare.

This project aims to bridge this gap by detecting news agencies in a large corpus of Swiss and Luxembourgish newspaper articles (the impresso corpus) for the years 1840-2000 using deep learning methods. For this, we first build and annotate a multilingual dataset with news agency mentions, which we then use to train and evaluate several BERT-based agency detection and classification models. Based on these experiments, we choose two models (for French and German) for the inference on the impresso corpus.


## Research Summary

Results show that ca. 10% of the articles explicitly reference news agencies, with the greatest share of agency content after 1940, although systematic citation of agencies already started slowly in the 1910s.
Differences in the usage of agency content across time, countries and languages as well as between newspapers reveal a complex network of news flows, whose exploration provides many opportunities for future work.

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
- `lib/`: Contains python scripts for the classification and the conversion of data (before and after annotation in Inception).
  - `bert-classification/`: Contains the text classification model as well as the code for its application on the _impresso_ corpus.
  - `inception_postprocessing/`
  - `inception_preprocessing/`
- `notebooks/`: Contains the notebooks used in the creation of the training corpus, annotation and analysis.
  - `1_sampling_training_data/`
  - `2_annotation/`
  - `3_classification/`
  - `4_analysis/`
- `report/`: Contains the report of the master project (Pdf and Zip for Latex).

## Installation and Usage

The project uses `python 3.10`. The dependencies for running the code can be found in `requirements.txt`. If only the classification is required, it suffices to install the dependencies specified in `lib/bert-classification/requirements_classification.txt`.


### Text Classification 

The model is based on `AutoModelForSequenceClassification` provided by the `transformers` library and it is a generic model class that will be instantiated as one of the sequence classification model classes of the library when created with the `from_pretrained(pretrained_model_name_or_path)` method.

The classification baseline has three modes:
- binary classification: is there any news agency mentioned in the text or not
- multiclass classification: a text can only belong to a news agency
- multiclass and multilabel classification: a text can belong to several news agencies

Additional to the in-model evaluation, the [HIPE-scorer](https://github.com/hipe-eval/HIPE-scorer) can be downloaded for evaluation on the task of named entity recognition. It provides the possibility to evaluate on different time periods and OCR-levels. In order for the latter to work, we changed ``row["MISC"]`` to ``row["OCR-INFO"]`` in line 200 of ``HIPE-scorer/hipe_evaluation/utils.py``.

## License

newsagency-classification - Lea Marxen    
Copyright (c) 2023 EPFL    
This program is licensed under the terms of the MIT. 
