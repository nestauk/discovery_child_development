"""
Evaluate the simple models ("log_regression", "knn", "random_forest", "sgd", "svm") and the distilbert-base-uncased models
on the test data (labelled data not involved in the training process).

Usage:

python discovery_child_development/pipeline/models/binary_classifier/evaluating_datasets/evaluate_test_data.py


"""
import pandas as pd
import numpy as np
import altair as alt
from discovery_child_development import PROJECT_DIR, binary_config, config, S3_BUCKET
from nesta_ds_utils.loading_saving import S3
from nesta_ds_utils.viz.altair import saving
from discovery_child_development.utils.huggingface_pipeline import (
    load_model,
    load_training_args,
    load_trained_model,
)
from discovery_child_development.getters.binary_classifier.gpt_labelled_datasets import (
    get_labelled_data_for_classifier,
)
from discovery_child_development.getters.binary_classifier.binary_classifier_model import (
    get_binary_classifier_models,
)
from discovery_child_development.utils.testing_examples_utils import (
    testing_examples_simple,
)
from discovery_child_development.utils.testing_examples_utils import (
    testing_examples_huggingface,
)
from discovery_child_development.utils.general_utils import replace_binary_labels

# Model vars
production = True

# Set the seed
SEED = config["seed"]
np.random.seed(SEED)

# Paths
S3_PATH = "models/binary_classifier/"
PATH_TO = f"{PROJECT_DIR}/outputs/data/models/"
MODEL_FILENAME = (
    f"gpt_labelled_binary_classifier_distilbert_production_{production}.tar.gz"
)

if __name__ == "__main__":
    # Loading the simple models
    models_simple = ["log_regression", "knn", "random_forest", "sgd", "svm"]
    models_all = {}
    for model in models_simple:
        # Save model to S3
        models_all[model] = S3.download_obj(
            bucket=S3_BUCKET,
            path_from=f"{S3_PATH}gpt_labelled_binary_classifier_{model}.pkl",
        )

    # Loading the distilbert-base-uncased model
    # Save the model locally
    get_binary_classifier_models(
        filename=MODEL_FILENAME, s3_path=S3_PATH, path_to=PATH_TO
    )

    # Load the model
    model_folder = (
        f"{PATH_TO}gpt_labelled_binary_classifier_distilbert_production_{production}"
    )
    model = load_model(model_path=model_folder, config=binary_config, num_labels=2)

    # Train model with early stopping
    training_args = load_training_args(**binary_config["training_args"])
    trainer = load_trained_model(
        model=model,
        args=training_args,
        config=binary_config,
    )

    # Get the labelled data
    test_data = get_labelled_data_for_classifier(set_type="test")
    test_data = replace_binary_labels(
        test_data, replace_cat=["Relevant", "Not-relevant"]
    )

    # Get the metrics for the results of the simple models
    results_df = pd.DataFrame()
    metrics = ["method", "accuracy", "precision", "recall", "f1"]
    for model in models_all:
        temp_df = pd.DataFrame(
            testing_examples_simple(
                list(test_data.text), list(test_data.labels), models_all[model]
            )[1],
            index=[0],
        )
        temp_df["method"] = model
        temp_df = temp_df[metrics]
        # Concat with results_df
        results_df = pd.concat([results_df, temp_df], axis=0)

    # Get the metrics for the results of the distilbert-base-uncased model
    results = testing_examples_huggingface(
        trainer, test_data[["labels", "text"]], binary_config
    )

    # Add Distilbert results to results_df
    temp_df = pd.DataFrame(
        {
            "method": "distilbert",
            "accuracy": results[1]["test_accuracy"],
            "precision": results[1]["test_precision"],
            "recall": results[1]["test_recall"],
            "f1": results[1]["test_f1"],
        },
        index=[0],
    )

    # Concat with results_df
    results_df = pd.concat([results_df, temp_df], axis=0).reset_index(drop=True)
    results_df["method_names"] = [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SGD",
        "SVM",
        "Distilbert",
    ]

    # Plotting the results
    # Create 4 bar charts for each metric
    # Accuracy
    accuracy_plot = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("method_names", title="Method", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("accuracy", title="Accuracy"),
        )
        .properties(width=200, height=200)
    )

    # Precision
    precision_plot = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("method_names", title="Method", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("precision", title="Precision"),
        )
        .properties(width=200, height=200)
    )

    # Recall
    recall_plot = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("method_names", title="Method", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("recall", title="Recall"),
        )
        .properties(width=200, height=200)
    )

    # F1
    f1_plot = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("method_names", title="Method", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("f1", title="F1"),
        )
        .properties(width=200, height=200)
    )

    # Combine all charts in a 2x2 grid
    metric_plot = alt.vconcat(
        accuracy_plot | precision_plot, recall_plot | f1_plot, data=results_df
    ).properties(title="Test data metrics")

    # Save the plot
    output_path = str(PROJECT_DIR) + "/outputs/figures/"
    saving.save(
        metric_plot,
        "binary_classifier_test_data_metrics",
        output_path,
        save_html=False,
        save_png=True,
    )
