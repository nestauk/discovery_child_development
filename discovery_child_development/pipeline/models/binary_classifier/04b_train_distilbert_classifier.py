"""
Run the binary classifier using distilbert-base-uncased.

Usage:

python discovery_child_development/pipeline/models/binary_classifier/04b_train_distilbert_classifier.py

Optional arguments:
--production : Determines whether to create the embeddings for the full dataset or a test sample (default: True)
--wandb : Determines whether a run gets logged on wandb (default: False)

"""
import wandb
import numpy as np
import argparse
from nesta_ds_utils.loading_saving import S3
from discovery_child_development.getters.binary_classifier.embeddings_hugging_face import (
    get_embeddings,
)
from discovery_child_development.utils.huggingface_pipeline import (
    load_model,
    load_training_args,
    load_trainer,
    saving_huggingface_model,
)
from discovery_child_development.utils.general_utils import replace_binary_labels
from discovery_child_development.utils import wandb as wb
from discovery_child_development.utils import classification_utils
from discovery_child_development.getters.binary_classifier.gpt_labelled_datasets import (
    get_labelled_data_for_classifier,
)
from discovery_child_development.utils.testing_examples_utils import (
    testing_examples_huggingface,
)
from discovery_child_development import (
    logging,
    S3_BUCKET,
    config,
    binary_config,
    PROJECT_DIR,
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
        default=True,
        help="Do you want to run the code in production? (default: True)",
    )

    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Do you want to log this as a run on wandb? (default: False)",
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

    if args.wandb:
        logging.info("Logging run on wandb")
        run = wandb.init(
            reinit=True,
            project="ISS supervised ML",
            job_type="Binary classifier - huggingface",
            save_code=True,
            tags=["gpt-labelled", "distilbert", "openealex/patents"],
        )

    # Load the model
    model = load_model(config=binary_config, num_labels=2)

    # Train model with early stopping
    training_args = load_training_args(output_dir=S3_PATH, config=binary_config)
    trainer = load_trainer(
        model=model,
        args=training_args,
        train_dataset=embeddings_training,
        eval_dataset=embeddings_validation,
        config=binary_config,
    )
    trainer.train()

    # Evaluate model
    trainer.evaluate()

    # View f1, roc and accuracy of predictions on validation set
    model_predictions = trainer.predict(embeddings_validation)

    logging.info(model_predictions.metrics)

    # Creating confusion matrix
    predictions = np.argmax(model_predictions.predictions, axis=-1)
    labels = model_predictions.label_ids.ravel().tolist()
    confusion_matrix = classification_utils.plot_confusion_matrix(
        labels, predictions, None, "Relevant works"
    )

    # Save model to S3
    SAVE_TRAINING_RESULTS_PATH = PROJECT_DIR / "outputs/data/models/"
    saving_huggingface_model(
        trainer,
        f"gpt_labelled_binary_classifier_distilbert_production_{args.production}",
        save_path=SAVE_TRAINING_RESULTS_PATH,
        s3_path=S3_PATH,
    )

    if args.wandb:
        # Log metrics
        wandb.run.summary["f1"] = model_predictions.metrics["test_f1"]
        wandb.run.summary["accuracy"] = model_predictions.metrics["test_accuracy"]
        wandb.run.summary["precision"] = model_predictions.metrics["test_precision"]
        wandb.run.summary["recall"] = model_predictions.metrics["test_recall"]

        # Adding reference to this model in wandb
        wb.add_ref_to_data(
            run=run,
            name=f"gpt_labelled_binary_classifier_distilbert_production_{args.production}",
            description=f"Distilbert model trained on binary classifier training data",
            bucket=S3_BUCKET,
            filepath=f"{S3_PATH}gpt_labelled_binary_classifier_distilbert_production_{args.production}.tar.gz",
        )

        # Log confusion matrix
        wb_confusion_matrix = wandb.Table(
            data=confusion_matrix, columns=confusion_matrix.columns
        )
        run.log({"confusion_matrix": wb_confusion_matrix})

        # End the weights and biases run
        wandb.finish()

    # Checking results by source
    validation_data = get_labelled_data_for_classifier(set_type="validation")
    openalex_val = replace_binary_labels(
        validation_data.query("source == 'openalex'"),
        replace_cat=["Relevant", "Not-relevant"],
    )
    patents_val = replace_binary_labels(
        validation_data.query("source == 'patents'"),
        replace_cat=["Relevant", "Not-relevant"],
    )

    # Get results
    for data, names in zip([openalex_val, patents_val], ["openalex", "patents"]):
        predictions, metrics = testing_examples_huggingface(
            trainer, data[["labels", "text"]], binary_config
        )
        if args.wandb:
            logging.info("Logging source breakdown on wandb")
            run = wandb.init(
                reinit=True,
                project="ISS supervised ML",
                job_type="Binary classifier - huggingface",
                save_code=True,
                tags=["gpt-labelled", "distilbert", "openealex/patents", names],
            )
            # Log metrics
            wandb.run.summary["f1"] = metrics["test_f1"]
            wandb.run.summary["accuracy"] = metrics["test_accuracy"]
            wandb.run.summary["precision"] = metrics["test_precision"]
            wandb.run.summary["recall"] = metrics["test_recall"]

            # Log confusion matrix
            wb_confusion_matrix = wandb.Table(
                data=confusion_matrix, columns=confusion_matrix.columns
            )
            run.log({f"confusion_matrix_{names}": wb_confusion_matrix})

            # End the weights and biases run
            wandb.finish()
