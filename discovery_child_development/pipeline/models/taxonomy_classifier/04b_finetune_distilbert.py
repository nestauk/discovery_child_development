"""
Usage:

python discovery_child_development/pipeline/models/taxonomy_classifier/04b_finetune_distilbert.py --production=False --wandb=True
"""
import argparse
import torch
import wandb

from discovery_child_development import taxonomy_config, logging
from discovery_child_development.getters.taxonomy_classifier import get_hf_ds
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import huggingface_pipeline as hf
from discovery_child_development.utils import utils

# Path to save intermediary training results and best model
SAVE_TRAINING_RESULTS_PATH = taxonomy_config["models_path"]
utils.create_directory_if_not_exists(SAVE_TRAINING_RESULTS_PATH)

S3_PATH = taxonomy_config["s3_models_path"]

THRESHOLD = taxonomy_config["predictions_threshold"]

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
        default=True,
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
            tags=["finetuning", f"production_{args.production}"],
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

    y_pred = hf.binarise_predictions(predictions.predictions, threshold=THRESHOLD)
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

    # Save model to S3
    logging.info("Saving model to S3...")
    hf.saving_huggingface_model(
        trainer,
        f"taxonomy_classifier_distilbert_production_{args.production}",
        save_path=SAVE_TRAINING_RESULTS_PATH,
        s3_path=S3_PATH,
    )
