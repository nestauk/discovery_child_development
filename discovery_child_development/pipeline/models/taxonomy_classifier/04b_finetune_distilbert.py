import argparse
import torch
import wandb

from nesta_ds_utils.loading_saving import S3

from discovery_child_development import PROJECT_DIR, S3_BUCKET, taxonomy_config, logging
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import huggingface_pipeline as hf

HF_PATH = taxonomy_config["s3_hf_ds_path"]
HF_FILE = taxonomy_config["s3_hf_ds_file"]

# Path to save intermediary training results and best model
SAVE_TRAINING_RESULTS_PATH = PROJECT_DIR / "outputs/data/models/"
SAVE_TRAINING_RESULTS_PATH.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--production",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Do you want to run the code in production? (default: False)",
    )

    parser.add_argument(
        "--wandb",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Do you want to run an experiment on weights and biases? (default: True)",
    )

    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    if args.wandb:
        run = wandb.init(
            project="ISS supervised ML",
            job_type="Taxonomy classifier",
            save_code=True,
            tags=["finetuning"],
        )

    if args.production == False:
        train_ds = get_hf_ds(set_type="train")
        val_ds = get_hf_ds(set_type="val")
    else:
        train_ds = get_hf_ds(set_type="train", production=True)
        val_ds = get_hf_ds(set_type="val", production=True)

    # label names need to be brought in later when evaluating the predictions
    label_names = [
        key
        for key in train_ds[0].keys()
        if key not in ["id", "source", "text", "labels", "input_ids", "attention_mask"]
    ]

    NUM_LABELS = len(label_names)

    # Load model
    model = hf.load_model(config=taxonomy_config, num_labels=NUM_LABELS)

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

    # Evaluate model
    trainer.evaluate()

    # View f1, roc and accuracy of predictions on validation set
    predictions = trainer.predict(val_ds)
    hf.compute_metrics_multilabel(predictions)

    y_pred = hf.binarise_predictions(predictions.predictions, threshold=0.5)
    y_pred_proba = torch.sigmoid(torch.Tensor(predictions.predictions)).numpy()
    y_true = predictions.label_ids

    metrics = classification_utils.create_average_metrics(
        y_true, y_pred, average="macro"
    )
    logging.info(metrics)

    confusion_matrix = classification_utils.create_confusion_matrix(
        y_true, y_pred, label_names, proportions=False
    )

    if args.wandb:
        wandb.run.summary["macro_avg_f1"] = metrics["f1"]
        wandb.run.summary["accuracy"] = metrics["accuracy"]
        wandb.run.summary["macro_avg_precision"] = metrics["precision"]
        wandb.run.summary["macro_avg_recall"] = metrics["recall"]

        # Log confusion matrix
        wb_confusion_matrix = wandb.Table(
            data=confusion_matrix, columns=confusion_matrix.columns
        )
        run.log({"confusion_matrix": wb_confusion_matrix})

        wandb.finish()
