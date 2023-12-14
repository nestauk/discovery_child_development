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


def load_tokenizer(config: dict, problem_type: bool = True) -> DistilBertTokenizerFast:
    """Load multi label classification BERT tokenzier

    Args:
        config: Dictionary containing model config

    Returns:
        BERT tokenizer
    """
    if problem_type:
        return AutoTokenizer.from_pretrained(
            "distilbert-base-uncased", problem_type=config["problem_type"]
        )
    else:
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_dataset(
    dataset: Dataset, text_column: str, config: dict, problem_type: bool = True
) -> Dataset:
    """Tokenize text in dataset"""
    remove_cols = dataset.column_names
    remove_cols.remove("labels")
    tokenizer = load_tokenizer(config=config, problem_type=problem_type)
    return dataset.map(
        lambda row: tokenizer(row[text_column], padding="max_length", truncation=True),
        batched=True,
        remove_columns=remove_cols,
    )


def df_to_hf_ds(
    df: pd.DataFrame,
    config: dict,
    non_label_cols: list = ["text", "id"],
    text_column: str = "text",
    problem_type: bool = True,
) -> Dataset:
    """Converts a dataframe into a huggingface dataset.
    Adds labels and tokenizes the text.

    Args:
        df: Dataframe to convert into a dataset
        config: Dictionary containing model config
        non_label_cols: Columns that are not labels. This not needed for binary classification.
            Defaults to ["text", "id"].
        text_column: Column in dataframe that contain text to tokenize.
            Defaults to "text".

    Returns:
        Huggingface dataset
    """
    dataset = Dataset.from_pandas(df, preserve_index=False)
    if problem_type:
        dataset = create_labels(dataset, cols_to_skip=non_label_cols)
    return tokenize_dataset(dataset, text_column=text_column, config=config)


def load_model(
    config: dict,
    num_labels: int = None,
    model_path: Union[str, Path] = "distilbert-base-uncased",
    problem_type: bool = True,
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
    if problem_type:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=num_labels,
            problem_type=config["problem_type"],
        )
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path, num_labels=num_labels
        )


def load_training_args(
    output_dir: Union[str, Path], config: dict
) -> transformers.training_args.TrainingArguments:
    """Load Training Arguments to be used to train the model

    Args:
        output_dir: Path to save training results
        config: Dictionary of training arguments

    Returns:
        TrainingArguments object
    """

    return TrainingArguments(
        output_dir=output_dir,
        report_to=config["report_to"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],
        metric_for_best_model=config["metric_for_best_model"],
        load_best_model_at_end=config["load_best_model_at_end"],
    )


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
    # flatten labels to list
    # labels = np.concatenate(labels).ravel().tolist()
    # print(labels)
    return metrics.compute(predictions=predictions, references=labels)


def load_trainer(
    model: DistilBertForSequenceClassification,
    args: transformers.training_args.TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: dict,
    problem_type: bool = True,
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
    if problem_type:
        compute_metrics = compute_metrics_multilabel
    else:
        compute_metrics = compute_metrics_binary

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=load_tokenizer(config=config, problem_type=problem_type),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"]
            )
        ],
    )


def saving_huggingface_model(
    trainer, output_filename: str, save_path: str, s3_path: str
):
    """Saves a huggingface model to S3

    Args:
        trainer (transformers.Trainer): Trained huggingface trainer
        output_filename (str): Name of the file to be saved
        save_path (str): Path to save the model to

    Returns:
        None: Saves the model locally and uploads to S3
    """
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
