from datasets import Dataset
import torch
import transformers
import evaluate
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
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import S3_BUCKET
from discovery_child_development.utils.general_utils import make_tarfile
from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# The following problem types are allowed in this classification pipeline: ["regression", "single_label_classification", "multi_label_classification"]


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


def load_tokenizer(config: dict) -> DistilBertTokenizerFast:
    """Load multi label classification BERT tokenzier

    Args:
        config: Dictionary containing model config

    Returns:
        BERT tokenizer
    """
    return AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", problem_type=config["problem_type"]
    )


def tokenize_dataset(
    dataset: Dataset,
    text_column: str,
    config: dict,
    prediction: bool = False,
) -> Dataset:
    """Tokenize text in dataset"""
    remove_cols = dataset.column_names
    if not prediction:
        remove_cols.remove("labels")
    tokenizer = load_tokenizer(config=config)
    return dataset.map(
        lambda row: tokenizer(row[text_column], padding="max_length", truncation=True),
        batched=True,
        remove_columns=remove_cols,
    )


def df_to_hf_ds(
    df: pd.DataFrame,
    config: dict,
    non_label_cols: list = ["text", "id", "source"],
    text_column: str = "text",
    prediction: bool = False,
) -> Dataset:
    """Converts a dataframe into a huggingface dataset.
    Adds labels and tokenizes the text.

    Args:
        df: Dataframe to convert into a dataset
        config: Dictionary containing model config
        non_label_cols: Columns that are not labels. This not needed for binary classification.
            Defaults to ["text", "id", "source"].
        text_column: Column in dataframe that contain text to tokenize.
            Defaults to "text".

    Returns:
        Huggingface dataset
    """
    dataset = Dataset.from_pandas(df, preserve_index=False)
    if config["problem_type"] == "multi_label_classification":
        dataset = create_labels(dataset, cols_to_skip=non_label_cols)

    if prediction:
        tokenized_data = tokenize_dataset(
            dataset, text_column=text_column, config=config, prediction=prediction
        )
    else:
        tokenized_data = tokenize_dataset(
            dataset, text_column=text_column, config=config
        )

    # Prepare the final dataset with only the required columns
    def select_required_columns(example, index):
        # Select only the required keys
        if prediction:
            required_keys = non_label_cols + ["input_ids", "attention_mask"]
        else:
            required_keys = non_label_cols + ["labels", "input_ids", "attention_mask"]
        return {
            key: example[key] if key in example else tokenized_data[index][key]
            for key in required_keys
        }

    final_dataset = dataset.map(select_required_columns, with_indices=True)

    return final_dataset


def load_model(
    config: dict,
    num_labels: int = None,
    model_path: Union[str, Path] = "distilbert-base-uncased",
) -> DistilBertForSequenceClassification:
    """Loads multi label BERT classifier

    Args:
        num_labels: Number of labels in the dataset. Defaults to None.
        config: Dictionary containing model config
        model_path: Defaults to "distilbert-base-uncased". Alternatively,
            can specify path to a fine tuned model.

    Returns:
        BERT classifier model
    """

    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=num_labels,
        problem_type=config["problem_type"],
    )


def load_training_args(
    **training_args: dict,
) -> transformers.training_args.TrainingArguments:
    """Load Training Arguments to be used to train the model

    Args:
        training_args: Dictionary of training arguments

    Returns:
        TrainingArguments object
    """

    return TrainingArguments(**training_args)


def binarise_predictions(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply sigmoid transformation to predictions and set
    values >= threshold to 1 and < threshold to 0"""
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    binarised = np.zeros(probs.shape)
    binarised[np.where(probs >= threshold)] = 1
    return binarised


def label_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    averaging: str = "macro",
) -> dict:
    """Calculate and return dictionary of metrics that are useful
    for measuring multi label, multi class or binary classification models"""
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


def compute_metrics_multilabel(model_predictions: EvalPrediction) -> dict:
    """Compute metrics in format suitable for pytorch training

    Args:
        model_predictions: Model predictions

    Returns:
        Dictionary of metrics
    """
    preds = (
        model_predictions.predictions[0]
        if isinstance(model_predictions.predictions, tuple)
        else model_predictions.predictions
    )

    return label_metrics(
        predictions=preds, labels=model_predictions.label_ids, averaging="macro"
    )


def compute_metrics_binary(model_predictions: EvalPrediction) -> dict:
    """Compute metrics in format suitable for pytorch training

    Args:
        model_predictions: Model predictions

    Returns:
        Dictionary of metrics
    """
    metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    logits, labels = model_predictions
    predictions = np.argmax(logits, axis=-1)

    return metrics.compute(predictions=predictions, references=labels)


def load_trainer(
    model: DistilBertForSequenceClassification,
    args: transformers.training_args.TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: dict,
) -> Trainer:
    """Load model trainer which can be used to train a model or make predictions

    Args:
        model: Model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Dictionary of training arguments

    Returns:
        Trainer object
    """
    if config["problem_type"] == "multi_label_classification":
        compute_metrics = compute_metrics_multilabel
    else:
        compute_metrics = compute_metrics_binary

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=load_tokenizer(config=config),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"]
            )
        ],
    )


def saving_huggingface_model(
    trainer: Trainer, output_filename: str, save_path: str, s3_path: str
):
    """Saves a huggingface model to S3

    Args:
        trainer (transformers.Trainer): Trained huggingface trainer
        output_filename (str): Name of the file to be saved
        save_path (str): Path to save the model to

    Returns:
        None: Saves the model locally and uploads to S3
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    # Saving the model locally as a folder
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = Path.joinpath(save_path, output_filename)
    trainer.save_model(model_path)
    # Converting folder to zipped file
    make_tarfile(str(model_path) + ".tar.gz", str(model_path))
    # Uploading to S3
    S3.upload_file(
        path_from=str(model_path) + ".tar.gz",
        bucket=S3_BUCKET,
        path_to=f"{s3_path}{output_filename}.tar.gz",
    )


def load_trained_model(
    model: DistilBertForSequenceClassification,
    args: transformers.training_args.TrainingArguments,
    config: dict,
) -> Trainer:
    """Load trained which can be used make predictions

    Args:
        model: Model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Dictionary of training arguments

    Returns:
        Trainer object
    """
    if config["problem_type"] == "multi_label_classification":
        compute_metrics = compute_metrics_multilabel
    else:
        compute_metrics = compute_metrics_binary

    return Trainer(
        model=model,
        args=args,
        tokenizer=load_tokenizer(config=config),
        compute_metrics=compute_metrics,
    )


def predictions_huggingface(
    trainer: Trainer, text_data: list, config: dict
) -> pd.DataFrame:
    """Output predictions from the huggingface classifier

    Args:
        trainer (transformers.Trainer): Trained huggingface trainer
        text_data (list): List of strings
        config (dict): Dictionary containing model config

    Returns:
        pd.DataFrame: Dataframe containing the text and predictions

    """
    # Convert to HF dataset
    text_data_df = df_to_hf_ds(
        text_data,
        config=config,
        non_label_cols=["text"],
        text_column="text",
        prediction=True,
    )

    # Predict the labels for the text data
    model_predictions = trainer.predict(text_data_df)
    # Binarise predictions
    predictions = np.argmax(model_predictions.predictions, axis=-1)

    text_data["predictions"] = predictions
    return text_data
