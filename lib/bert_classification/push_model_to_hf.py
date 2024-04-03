from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
from newsagency_ner import (
    NewsAgencyModelPipeline,
)
from model import (
    ModelForSequenceAndTokenClassification,
)
from huggingface_hub import HfApi, HfFolder

api = HfApi()


label2id = {
    "B-org.ent.pressagency.Reuters": 0,
    "B-org.ent.pressagency.Stefani": 1,
    "O": 2,
    "B-org.ent.pressagency.Extel": 3,
    "B-org.ent.pressagency.Havas": 4,
    "I-org.ent.pressagency.Xinhua": 5,
    "I-org.ent.pressagency.Domei": 6,
    "B-org.ent.pressagency.Belga": 7,
    "B-org.ent.pressagency.CTK": 8,
    "B-org.ent.pressagency.ANSA": 9,
    "B-org.ent.pressagency.DNB": 10,
    "B-org.ent.pressagency.Domei": 11,
    "I-pers.ind.articleauthor": 12,
    "I-org.ent.pressagency.Wolff": 13,
    "B-org.ent.pressagency.unk": 14,
    "I-org.ent.pressagency.Stefani": 15,
    "I-org.ent.pressagency.AFP": 16,
    "B-org.ent.pressagency.UP-UPI": 17,
    "I-org.ent.pressagency.ATS-SDA": 18,
    "I-org.ent.pressagency.unk": 19,
    "B-org.ent.pressagency.DPA": 20,
    "B-org.ent.pressagency.AFP": 21,
    "I-org.ent.pressagency.DNB": 22,
    "B-pers.ind.articleauthor": 23,
    "I-org.ent.pressagency.UP-UPI": 24,
    "B-org.ent.pressagency.Kipa": 25,
    "B-org.ent.pressagency.Wolff": 26,
    "B-org.ent.pressagency.ag": 27,
    "I-org.ent.pressagency.Extel": 28,
    "I-org.ent.pressagency.ag": 29,
    "B-org.ent.pressagency.ATS-SDA": 30,
    "I-org.ent.pressagency.Havas": 31,
    "I-org.ent.pressagency.Reuters": 32,
    "B-org.ent.pressagency.Xinhua": 33,
    "B-org.ent.pressagency.AP": 34,
    "B-org.ent.pressagency.APA": 35,
    "I-org.ent.pressagency.ANSA": 36,
    "B-org.ent.pressagency.DDP-DAPD": 37,
    "I-org.ent.pressagency.TASS": 38,
    "I-org.ent.pressagency.AP": 39,
    "B-org.ent.pressagency.TASS": 40,
    "B-org.ent.pressagency.Europapress": 41,
    "B-org.ent.pressagency.SPK-SMP": 42,
}

id2label = {v: k for k, v in label2id.items()}


from transformers.pipelines import PIPELINE_REGISTRY


from huggingface_hub import Repository


def push_model_to_hub(model_dir, model_id):
    # Load the model
    model = ModelForSequenceAndTokenClassification.from_pretrained(
        model_dir,
        num_sequence_labels=2,
        num_token_labels=len(label2id),
    )
    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_dir,
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.config.id2label = id2label
    model.config.label2id = label2id

    model.config._name_or_path = model_id

    # Push to hub
    model.push_to_hub(model_id)
    tokenizer.push_to_hub(model_id)

    # Register your custom pipeline
    PIPELINE_REGISTRY.register_pipeline(
        "newsagency-ner",
        pipeline_class=NewsAgencyModelPipeline,
        pt_model=ModelForSequenceAndTokenClassification,
    )
    model.config.custom_pipelines = {
        "newsagency-ner": {
            "impl": "newsagency_ner.NewsAgencyModelPipeline",
            "pt": ["newsagency_ner.ModelForSequenceAndTokenClassification"],
            "tf": [],
        }
    }

    classifier = pipeline("newsagency-ner", model=model, tokenizer=tokenizer)
    # classifier = NewsAgencyModelPipeline(model=model, tokenizer=tokenizer)

    # Dynamically load the custom model class from the Hugging Face Hub
    # config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_id, config=config, trust_remote_code=True
    # )
    #
    # # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # classifier = pipeline("newsagency-ner", model=model, tokenizer=tokenizer)
    #
    # import pdb
    #
    # pdb.set_trace()
    # # classifier = pipeline("newsagency-ner", model=model_id)
    #
    print(classifier("Mon nom est François et j'habite à Paris. (Reuter)"))

    # Save your model and tokenizer in the local directory
    model.save_pretrained("bert-newsagency-ner-fr", config=model.config)
    tokenizer.save_pretrained("bert-newsagency-ner-fr")
    classifier.save_pretrained("bert-newsagency-ner-fr")
    # # repo.git_add()  # Add changes
    # # repo.git_commit("Update model and pipeline")  # Commit changes
    # # repo.push_to_hub()
    api.upload_folder(
        token=HfFolder.get_token(),
        folder_path="bert-newsagency-ner-fr",
        path_in_repo="",
        repo_id="impresso-project/bert-newsagency-ner-fr",
    )


# Directories where the models are saved
agency_fr_dir = "trained_models/agency-fr"
# agency_de_dir = "trained_models/agency-de"

# Model IDs on Hugging Face Hub (you can customize these)
agency_fr_model_id = "impresso-project/bert-newsagency-ner-fr"
# agency_de_model_id = "impresso-project/bert-newsagency-ner-de"

# Push the models to the Hugging Face Hub
push_model_to_hub(agency_fr_dir, agency_fr_model_id)
# push_model_to_hub(agency_de_dir, agency_de_model_id)