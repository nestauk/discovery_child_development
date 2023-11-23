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

# %%
# preamble


import numpy as np
import os
import pandas as pd


from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging, S3_BUCKET, config
from discovery_child_development.getters import openalex as oa
from discovery_child_development.pipeline.models.taxonomy_classifier import (
    baseline_model as bm,
)
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import cluster_analysis_utils as cau
from discovery_child_development.utils import wandb as wb


CONCEPT_IDS = "|".join(config["openalex_concepts"])
INPUT_PATH = f"data/openAlex/processed/openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_train.csv"
VECTORS_FILEPATH = "data/openAlex/vectors/sentence_vectors_384.parquet"
DATA_PATH_LOCAL = PROJECT_DIR / "inputs/data/"
FIG_PATH = PROJECT_DIR / "outputs/figures/"
MODEL_PATH = PROJECT_DIR / "outputs/models/"
SEED = 42
# Set the seed
np.random.seed(SEED)
random.seed(SEED)

NUM_SAMPLES = 1000


# %%
### fine-tuning utils ###


def create_labels(dataset: Dataset, cols_to_skip: list) -> Dataset:
    """Add labels to the dataset. Labels are a list of 0.0s and 1.0s
    corresponding to the order of the labels in the original dataframe
    that was used to create the dataset. Note the 0.0s and 1.0s must be floats."""
    cols = dataset.column_names
    return dataset.map(
        lambda row: {
            "labels": torch.FloatTensor(
                [(row[col]) for col in cols if col not in cols_to_skip]
            )
        }
    )


def load_tokenizer() -> DistilBertTokenizerFast:
    """Load multi label classification BERT tokenzier"""
    return AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", problem_type="multi_label_classification"
    )


def tokenize_dataset(dataset: Dataset, text_column: str) -> Dataset:
    """Tokenize text in dataset"""
    remove_cols = dataset.column_names
    remove_cols.remove("labels")
    tokenizer = load_tokenizer()
    return dataset.map(
        lambda row: tokenizer(row[text_column], padding="max_length", truncation=True),
        batched=True,
        remove_columns=remove_cols,
    )


def df_to_hf_ds(
    df: pd.DataFrame, non_label_cols: list = ["text", "id"], text_column: str = "text"
) -> Dataset:
    """Converts a dataframe into a huggingface dataset.
    Adds labels and tokenizes the text.

    Args:
        df: Dataframe to convert into a dataset
        non_label_cols: Columns that are not labels.
            Defaults to ["text", "id"].
        text_column: Column in dataframe that contain text to tokenize.
            Defaults to "text".

    Returns:
        Huggingface dataset
    """
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = create_labels(dataset, cols_to_skip=non_label_cols)
    return tokenize_dataset(dataset, text_column=text_column)


def load_model(
    num_labels: int, model_path: Union[str, Path] = "distilbert-base-uncased"
) -> DistilBertForSequenceClassification:
    """Loads multi label BERT classifier

    Args:
        num_labels: Number of labels
        model_path: Defaults to "distilbert-base-uncased". Alternatively,
            can specify path to a fine tuned model.

    Returns:
        BERT classifier model
    """
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )


def load_training_args(
    output_dir: Union[str, Path]
) -> transformers.training_args.TrainingArguments:
    """Load Training Arguments to be used to train the model"""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
    )


def binarise_predictions(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply sigmoid transformation to predictions and set
    values >= threshold to 1 and < threshold to 0"""
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    binarised = np.zeros(probs.shape)
    binarised[np.where(probs >= threshold)] = 1
    return binarised


def multi_label_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    averaging: str = "macro",
) -> dict:
    """Calculate and return dictionary of metrics that are useful
    for measuring multi label, multi class classification models"""
    y_pred = binarise_predictions(predictions, threshold)
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average=averaging)
    try:
        roc_auc = roc_auc_score(y_true, y_pred, average=averaging)
    except ValueError:
        roc_auc = float(
            "nan"
        )  # needed for testing on small samples where you are not guaranteed to have both positive and negative samples for all classes
    accuracy = accuracy_score(y_true, y_pred)
    return {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}


def compute_metrics(model_predictions: EvalPrediction) -> dict:
    """Compute metrics in format suitable for pytorch training"""
    preds = (
        model_predictions.predictions[0]
        if isinstance(model_predictions.predictions, tuple)
        else model_predictions.predictions
    )
    return multi_label_metrics(
        predictions=preds, labels=model_predictions.label_ids, averaging="macro"
    )


def load_trainer(
    model: DistilBertForSequenceClassification,
    args: transformers.training_args.TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    """Load model trainer which can be used to train a model or make predictions"""
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=load_tokenizer(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )


# %%
# Load the data. Just the training set by default
openalex_data, training_file_name = oa.get_labelled_data()
logging.info(training_file_name)

# Filter the data using a score threshold (0.3 is the threshold used by OpenAlex)
openalex_data_wide = (
    # Squash sub-categories into one tuple per work (rather than one row per sub-category per work).
    # This is the required input to the sklearn MultiLabelBinarizer.
    openalex_data.groupby(["openalex_id", "text"])["sub_category"]
    .agg(lambda x: tuple(set(x)))
    .reset_index()
)
# Set the index - useful later for creating training/validation split
openalex_data_wide = openalex_data_wide.set_index("openalex_id")

# Load embeddings
embeddings = S3.download_obj(
    S3_BUCKET,
    path_from=VECTORS_FILEPATH,
    download_as="dataframe",
)

embeddings = embeddings.set_index("openalex_id")

openalex_data_wide = openalex_data_wide.join(embeddings, on="openalex_id", how="left")

# The multilabel binarizer splits the sub-category tuple into binary labels.
# Y has a column for each unique sub-category in the data, and one row per OpenAlex ID.
Y, mlb = classification_utils.add_binarise_labels(
    openalex_data_wide, label_column="sub_category", not_valid_label=None
)

# Split IDs into random train and validation subsets
unique_ids = openalex_data_wide.index.unique()

# We will only get metrics on the training set for now, because the baseline should be
# the best possible score we can get from a probability/majority-based dummy classifier,
# and we assume the metrics will be slightly better on the training set.
train_ids, val_ids = train_test_split(unique_ids, test_size=0.1, random_state=SEED)

# just a sample to try fine-tuning
embeddings_train_ids = random.sample(list(train_ids), NUM_SAMPLES)
embeddings_val_ids = random.sample(list(val_ids), NUM_SAMPLES)

X_train = openalex_data_wide[openalex_data_wide.index.isin(embeddings_train_ids)][
    ["text"]
]
X_val = openalex_data_wide[openalex_data_wide.index.isin(embeddings_val_ids)][["text"]]

Y_train = Y[Y.index.isin(embeddings_train_ids)]
Y_val = Y[Y.index.isin(embeddings_val_ids)]

# %%
Y_train = Y_train.reset_index(names=["openalex_id"])
X_train = X_train.reset_index(names=["openalex_id"])

Y_val = Y_val.reset_index(names=["openalex_id"])
X_val = X_val.reset_index(names=["openalex_id"])

# %%
train_dataset_df = pd.merge(X_train, Y_train, on="openalex_id")
val_dataset_df = pd.merge(X_val, Y_val, on="openalex_id")

train_dataset_df.head()

# %%
train_ds = df_to_hf_ds(
    train_dataset_df, non_label_cols=["openalex_id", "text"], text_column="text"
)
val_ds = df_to_hf_ds(
    val_dataset_df, non_label_cols=["openalex_id", "text"], text_column="text"
)

# %%
# Set number of labels
NUM_LABELS = len(train_ds[0]["labels"])

# %%
# Path to save intermediary training results and best model
SAVE_TRAINING_RESULTS_PATH = PROJECT_DIR / "outputs/data/models/"
SAVE_TRAINING_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Load model
model = load_model(num_labels=NUM_LABELS)

# %%
# Train model with early stopping
training_args = load_training_args(output_dir=SAVE_TRAINING_RESULTS_PATH)
trainer = load_trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds
)
trainer.train()

# %%
# Evaluate model
trainer.evaluate()

# %%
# View f1, roc and accuracy of predictions on validation set
predictions = trainer.predict(val_ds)
compute_metrics(predictions)

# %%

# %%
