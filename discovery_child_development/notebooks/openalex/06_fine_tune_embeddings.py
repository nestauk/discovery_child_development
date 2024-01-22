# %%
from datasets import Dataset
import torch
import transformers
from transformers import (
    EvalPrediction,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.models.distilbert.tokenization_distilbert_fast import (
    DistilBertTokenizerFast,
)
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)
from typing import Union
from pathlib import Path
import random
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import wandb

# %%
PRODUCTION = False  # If false, the code will run on just a sample

# %%
from sklearn.model_selection import train_test_split
from typing import Union

import numpy as np
import pandas as pd

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import (
    PROJECT_DIR,
    logging,
    S3_BUCKET,
    config,
    taxonomy_config,
)
from discovery_child_development.getters import openalex as oa
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import huggingface_pipeline as hf

DATA_PATH_LOCAL = PROJECT_DIR / "inputs/data/"
FIG_PATH = PROJECT_DIR / "outputs/figures/"
MODEL_PATH = PROJECT_DIR / "outputs/models/"
SEED = config["seed"]
# Set the seed
np.random.seed(SEED)
random.seed(SEED)


# %%
run = wandb.init(
    project="ISS supervised ML",
    job_type="Taxonomy classifier",
    save_code=True,
    tags=["finetuning"],
)

# %%
HF_PATH = taxonomy_config["s3_hf_ds_path"]
HF_FILE = taxonomy_config["s3_hf_ds_file"]


# Load datasets
def get_hf_ds(
    s3_bucket=S3_BUCKET,
    s3_path=HF_PATH,
    s3_file=HF_FILE,
    set_type="train",
    production=False,
):
    if production:
        filepath = f"{s3_path}{s3_file.replace('SPLIT', set_type)}"
    else:
        filepath = f"{s3_path}test_{s3_file.replace('SPLIT', set_type)}"

    hf_ds = S3.download_obj(s3_bucket, path_from=filepath)

    return hf_ds


# %%
if PRODUCTION == False:
    train_ds = get_hf_ds(set_type="train")
    val_ds = get_hf_ds(set_type="val")
else:
    train_ds = get_hf_ds(set_type="train", production=True)
    val_ds = get_hf_ds(set_type="val", production=True)

# %%
train_ds[0].keys()

# %%
label_names = [
    key
    for key in train_ds[0].keys()
    if key not in ["id", "source", "text", "labels", "input_ids", "attention_mask"]
]
label_names

# %%
len(label_names)

# %%
# Set number of labels
NUM_LABELS = len(train_ds[3]["labels"])
NUM_LABELS

# %%
# Path to save intermediary training results and best model
SAVE_TRAINING_RESULTS_PATH = PROJECT_DIR / "outputs/data/models/"
SAVE_TRAINING_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Load model
model = hf.load_model(config=taxonomy_config, num_labels=NUM_LABELS)

# %%
# Train model with early stopping
training_args = hf.load_training_args(
    output_dir=SAVE_TRAINING_RESULTS_PATH, config=taxonomy_config
)
trainer = hf.load_trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    config=taxonomy_config,
)
trainer.train()

# %%
# Evaluate model
trainer.evaluate()

# %%
# View f1, roc and accuracy of predictions on validation set
predictions = trainer.predict(val_ds)
hf.compute_metrics_multilabel(predictions)

# %%
predictions

# %%
y_pred = hf.binarise_predictions(predictions.predictions, threshold=0.5)
y_pred_proba = torch.sigmoid(torch.Tensor(predictions.predictions)).numpy()
y_true = predictions.label_ids

# %%
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
print(f"Precision: {precision}; recall: {recall}")

# %%
# get the metrics for individual labels
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
print(f"Precision: {precision}; recall: {recall}")

# %%
confusion_matrix = classification_utils.create_confusion_matrix(
    y_true, y_pred, label_names, proportions=False
)

# %%
classification_utils.create_heatmap_table(
    y_true, y_pred, label_names, proportions=False
)

# %%
metrics = classification_utils.create_average_metrics(y_true, y_pred, average="macro")
metrics

# %%
wandb.run.summary["macro_avg_f1"] = metrics["f1"]
wandb.run.summary["accuracy"] = metrics["accuracy"]
wandb.run.summary["macro_avg_precision"] = metrics["precision"]
wandb.run.summary["macro_avg_recall"] = metrics["recall"]

# %%
# Log confusion matrix
wb_confusion_matrix = wandb.Table(
    data=confusion_matrix, columns=confusion_matrix.columns
)
run.log({"confusion_matrix": wb_confusion_matrix})

# %%
wandb.finish()

# %%
