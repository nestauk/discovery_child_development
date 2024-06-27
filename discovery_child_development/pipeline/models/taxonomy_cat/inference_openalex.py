"""
Script to do inference on the complete dataset with a trained classifier

"""
import pandas as pd
import argparse
import json
from discovery_child_development import PROJECT_DIR, S3_BUCKET, logging
from nesta_ds_utils.loading_saving import S3

from discovery_child_development.getters.openalex import get_sentence_embeddings

# Path to models
from discovery_child_development.pipeline.models.taxonomy_cat.train_classifiers import (
    MODEL_PATH,
    S3_MODEL_PATH,
    MODELS_SIMPLE,
)

# Path to data to be labelled
ENRICHED_DATA_DIR = PROJECT_DIR / "outputs/enrichments"
PATH_TO_DATASET = (
    ENRICHED_DATA_DIR / "openalex_relevance_labels_only_relevant.csv"
)

# Path to sentence embeddings
VECTORS_PATH = "data/outputs/vectors/"
VECTORS_FILE = "sentence_vectors_openalex_384_labelled.parquet"

PATH_TO_TOPICS = (
    PROJECT_DIR
    / "discovery_child_development/pipeline/labelling/taxonomy_cat/prompts/topics.json"
)

def inference_simple(examples: pd.DataFrame, classifier):
    """Testing the simple classifier on some examples

    Args:
        examples (list): List of strings
        labels (list): List of labels
        classifier (sklearn classifier): Trained classifier

    Returns:
        predictions (list): List of predictions
        metrics (dict): Dictionary of metrics
    """
    test_df = examples.copy()
    X_test = test_df["miniLM_384_vector"].apply(pd.Series).values
    predictions = classifier.predict(X_test)
    try:
        probs = classifier.predict_proba(X_test)[:, 1]
    except:
        probs = None
    return test_df.assign(labels=predictions).assign(prob_relevant=probs)


def parse_arguments():
    """Parse the arguments passed to the script."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the arguments for the script")
    # Add the arguments, and define defaults
    parser.add_argument("--topic", type=str, help="Taxonomy category", default="ai2")
    # Parse the arguments
    return parser.parse_args()


if __name__ == "__main__":
    # Define the arguments
    # args = parse_arguments()
    # try:
    #     topic = args.topic
    # except ValueError:
    #     raise ValueError("You must provide a valid topic")
    topics_dict = json.load(open(PATH_TO_TOPICS, "r"))
    topics = list(topics_dict.keys())

    # Load dataset sentence embeddings (all-MiniLM-L6-v2)
    embeddings_all = (
        get_sentence_embeddings(
            s3_bucket=S3_BUCKET,
            filepath=VECTORS_PATH,
            filename=VECTORS_FILE,
            id="id",
        )
        .reset_index()
        # Simplify the id by removing https
        .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
        .set_index("id")
    )
    # Load in the data that's labelled as relevant
    relevant_df = (
        pd.read_csv(PATH_TO_DATASET)
        .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
        .merge(embeddings_all.reset_index(), on="id", how="left")
    )
    for topic in topics:
        # Load all the models
        logging.info(f"Inference for topic {topic}")
        
        models_all = {}
        for model in MODELS_SIMPLE:
            model_path = f"{S3_MODEL_PATH}taxonomy_cat_classifier_{topic}_{model}.pkl"
            models_all[model] = S3.download_obj(bucket=S3_BUCKET, path_from=model_path)
        # Apply all the models on the dataset
        results_model = []
        for model in models_all:
            results_df = (
                inference_simple(relevant_df, models_all[model])
                .rename(columns={"labels": "prediction"})
                .assign(model=model)
            )[["id", "prediction", "prob_relevant", "model"]]
            results_model.append(results_df)
        results_model = pd.concat(results_model, axis=0)
        # Get ensemble results
        ensemble_df = results_model.groupby(["id"]).agg(
            prediction=("prediction", "mean"),
            prob_relevant=("prob_relevant", "mean"),
        )
        # Change results_model to long dataframe format
        results_model_long_df = (
            results_model.drop(columns=["prob_relevant"])
            .pivot(index="id", columns="model", values="prediction")
            .reset_index()
        )

        # Create the final dataframe
        results_df = (
            relevant_df.merge(ensemble_df, on="id", how="left")
            .merge(results_model_long_df, on="id", how="left")
            .drop(columns=["miniLM_384_vector"])
        )
        # Export the final data frame
        (
            results_df[["id", "prediction", "prob_relevant"] + list(models_all.keys())]
            .assign(topic=topic)
            .to_csv(
                ENRICHED_DATA_DIR
                / f"taxonomy_cat/openalex/taxonomy_cat_predictions_{topic}.csv",
                index=False,
            )
        )