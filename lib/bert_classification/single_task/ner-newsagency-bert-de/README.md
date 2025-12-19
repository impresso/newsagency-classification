---
language:
- de
license: agpl-3.0
---

## About

Since their beginnings in the 1830s and 1840s, news agencies have played an important role in the national and international news market, aiming to deliver news as fast and as reliable as possible. While we know that newspapers have been using agency content for a long time to produce their stories, the amount to which the agencies shape our news often remains unclear. Although researchers have already addressed this question, recently by using computational methods to assess the influence of news agencies at present, large-scale studies on the role of news agencies in the past continue to be rare.

This project aimed at bridging this gap by detecting news agencies in a large corpus of Swiss and Luxembourgish newspaper articles (the [impresso](https://impresso-project.ch/) corpus) for the years 1840-2000 using deep learning methods. For this, we first build and annotate a multilingual dataset with news agency mentions, which we then use to train and evaluate several BERT-based agency detection and classification models. Based on these experiments, we choose two models (for French and German) for the inference on the [impresso](https://impresso-project.ch/) corpus.

## Model Details

The base of the model is [dbmdz/bert-base-french-europeana-cased](https://huggingface.co/dbmdz/bert-base-french-europeana-cased) finetuned for 3 epochs on text of 256 maximum length.

## Research Summary

Results show that ca. 10% of the articles explicitly reference news agencies, with the greatest share of agency content after 1940, although systematic citation of agencies already started slowly in the 1910s.
Differences in the usage of agency content across time, countries and languages as well as between newspapers reveal a complex network of news flows, whose exploration provides many opportunities for future work.

## Dataset Characteristics
The dataset contains 1,133 French and 397 German annotated documents, with 1,058,449 tokens, of which 1,976 have annotations. Below is an overview of the corpus statistics:
The annotated dataset is released on [Zenodo](https://doi.org/10.5281/zenodo.8333933). 


Overview of corpus statistics. %noisy gives the percentage of agency mentions with at least one OCR error.

| Lg.   | Docs | Tokens  | Mentions | %noisy |
|-------|------|---------|----------|--------|
| Train | de   | 333     | 247,793  | 493    | 9%     |
|       | fr   | 903     | 606,671  | 1,122  | 5%     |
| Total |      | 1,236   | 854,464  | 1,615  | 6%     |
| Dev   | de   | 32      | 28,745   | 26     | 8%     |
|       | fr   | 110     | 77,746   | 114    | 3%     |
| Total |      | 142     | 106,491  | 140    | 4%     |
| Test  | de   | 32      | 22,437   | 58     | 3%     |
|       | fr   | 120     | 75,057   | 163    | 7%     |
| Total |      | 152     | 97,494   | 221    | 6%     |
| All   | de   | 397     | 298,975  | 577    | 9%     |
|       | fr   | 1,133   | 759,474  | 1,399  | 5%     |
| Total |      | 1,530   | 1,058,449| 1,976  | 6%     |


#### How to use

You can use this model with Transformers *pipeline* for NER.

```python
from transformers import pipeline

nlp = pipeline("newsagency-ner", model="impresso-project/bert-newsagency-ner-de", trust_remote_code=True)
nlp("Mein Name ist Wolfgang und ich wohne in Berlin. (Havas)")
```

### BibTeX entry and citation info

The code is available [here](https://github.com/impresso/newsagency-classification/tree/main).

```
@misc{marxen_newsagency_2023,
  title = "Where Did the News come from? Detection of News Agency Releases in Historical Newspapers",
  author = "Marxen, Lea and Ehrmann, Maud and Boros, Emanuela",
  year = "2023",
  url = "https://github.com/impresso/newsagency-classification/",
  note = "Master Thesis"
}
```