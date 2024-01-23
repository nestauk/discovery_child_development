"""
Run the binary classifier using distilbert-base-uncased.

Usage:

python discovery_child_development/pipeline/models/binary_classifier/04b_train_distilbert_classifier_sweeps.py

Optional arguments:
--production : Determines whether to create the embeddings for the full dataset or a test sample (default: True)

"""
import wandb
import numpy as np
import argparse
from nesta_ds_utils.loading_saving import S3
from transformers import TrainingArguments
from discovery_child_development.getters.binary_classifier.embeddings_hugging_face import (
    get_embeddings,
)
from discovery_child_development.utils.huggingface_pipeline import (
    load_model,
    load_trainer,
)
from discovery_child_development.utils import wandb as wb
from discovery_child_development import (
    logging,
    S3_BUCKET,
    config,
    binary_config,
)
from transformers import set_seed

# Set up
S3_PATH = "models/binary_classifier/"
VECTORS_PATH = "data/labels/binary_classifier/vectors/"
VECTORS_FILE = "distilbert_sentence_vectors_384_labelled"
SEED = config["seed"]
# Set the seed
set_seed(SEED)

if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--production",
        type=bool,
        default=False,
        help="Do you want to run the code in production? (default: False)",
    )
    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    if not args.production:
        VECTORS_FILE = VECTORS_FILE + "_test"

    # Loading the training and validation embeddings
    embeddings_training = get_embeddings(
        identifier="",
        production=args.production,
        set_type="train",
        vectors_path=VECTORS_PATH,
        vectors_file=VECTORS_FILE,
    )
    embeddings_validation = get_embeddings(
        identifier="",
        production=args.production,
        set_type="validation",
        vectors_path=VECTORS_PATH,
        vectors_file=VECTORS_FILE,
    )

    # Set up sweep config
    sweep_config = binary_config["sweep_config"]

    logging.info("Logging run on wandb")
    sweep_id = wandb.sweep(sweep_config, project="ISS supervised ML")

    # Load the model
    model = load_model(config=binary_config, num_labels=2)

    def sweep_training(config=None):
        with wandb.init(config=config):
            # Set up the config
            config = wandb.config
            # Train model with early stopping
            training_args = TrainingArguments(
                output_dir=S3_PATH,
                report_to=binary_config["report_to"],
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=binary_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=binary_config[
                    "gradient_accumulation_steps"
                ],
                num_train_epochs=config.num_train_epochs,
                weight_decay=config.weight_decay,
                evaluation_strategy=binary_config["evaluation_strategy"],
                save_strategy=binary_config["save_strategy"],
                metric_for_best_model=binary_config["metric_for_best_model"],
                load_best_model_at_end=binary_config["load_best_model_at_end"],
                seed=binary_config["seed"],
            )

            trainer = load_trainer(
                model=model,
                args=training_args,
                train_dataset=embeddings_training,
                eval_dataset=embeddings_validation,
                config=binary_config,
            )

            trainer.train()

    wandb.agent(sweep_id, sweep_training, count=20)
