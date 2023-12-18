"""
[WORK IN PROGRESS]
Run the binary classifier using distilbert-base-uncased.

Usage:

python discovery_child_development/pipeline/openalex/binary_classifier/05b_train_distilbert_classifier.py

Optional arguments:
--production : Determines whether to create the embeddings for the full dataset or a test sample (default: True)
--wandb : Determines whether a run gets logged on wandb (default: False)
--identifier : Choose which split of the training data you want (default: 50, 50/50 relevant/non-relevant). Options are "20", "50", "all".

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
from discovery_child_development.utils import wandb as wb
from discovery_child_development.utils import classification_utils
from discovery_child_development import (
    logging,
    S3_BUCKET,
    config,
    binary_config,
    PROJECT_DIR,
)

# Set up
S3_PATH = "models/binary_classifier/"
SEED = config["seed"]
NUM_SAMPLES = config["embedding_sample_size"]
# Set the seed
np.random.seed(SEED)

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

    parser.add_argument(
        "--identifier",
        type=str,
        default="50",
        help="Choose which split of the training data you want (default: 50, 50/50 relevant/non-relevant)",
    )
    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    # Loading the training and validation embeddings
    embeddings_training = get_embeddings(
        identifier=args.identifier, production=args.production, set_type="train"
    )
    embeddings_validation = get_embeddings(
        identifier=args.identifier, production=args.production, set_type="validation"
    )

    if args.wandb:
        logging.info("Logging run on wandb")
        run = wandb.init(
            project="ISS supervised ML",
            job_type="Binary classifier - huggingface",
            save_code=True,
            tags=["huggingface", "binary_classifier", "sentence_embeddings"],
        )

    # Load the model
    model = load_model(onfig=binary_config, num_labels=2)

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
        f"binary_classifier_distilbert_{args.identifier}_production_{args.production}",
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
            name=f"binary_classifier_{model}_" + args.identifier,
            description=f"{model} model trained on binary classifier training data",
            bucket=S3_BUCKET,
            filepath=f"{S3_PATH}binary_classifier_{model}_{args.identifier}.pkl",
        )

        # Log confusion matrix
        wb_confusion_matrix = wandb.Table(
            data=confusion_matrix, columns=confusion_matrix.columns
        )
        run.log({"confusion_matrix": wb_confusion_matrix})

        # End the weights and biases run
        wandb.finish()
