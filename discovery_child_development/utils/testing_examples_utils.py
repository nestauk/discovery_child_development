import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils.general_utils import replace_binary_labels
from discovery_child_development.utils.huggingface_pipeline import df_to_hf_ds


def testing_examples_simple(examples: list, labels: list, classifier):
    """Testing the simple classifier on some examples

    Args:
        examples (list): List of strings
        labels (list): List of labels
        classifier (sklearn classifier): Trained classifier

    Returns:
        predictions (list): List of predictions
        metrics (dict): Dictionary of metrics
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_vectors_384 = model.encode(examples, show_progress_bar=True)
    vectors_as_list = [list(vec) for vec in sentence_vectors_384]

    test_df = pd.DataFrame(
        {
            "openalex_id": [str(i) for i in range(len(vectors_as_list))],
            "miniLM_384_vector": vectors_as_list,
        }
    )
    test_df["label"] = labels

    # Setting up the test set
    X_test = test_df["miniLM_384_vector"].apply(pd.Series).values
    Y_test = test_df["label"]

    predictions = classifier.predict(X_test)

    # Creating metrics
    metrics = classification_utils.create_average_metrics(
        Y_test, predictions, average="binary"
    )

    return predictions, metrics


def testing_examples_huggingface(trainer, examples, config):
    """Testing the huggingface classifier on some examples

    Args:
        trainer (transformers.Trainer): Trained huggingface trainer
        examples (list): List of strings
        config (dict): Dictionary containing model config

    Returns:
        predictions (list): List of predictions
        metrics (dict): Dictionary of metrics
    """

    # Binarise labels
    examples = replace_binary_labels(examples, replace_cat=["Relevant", "Not relevant"])
    # Convert to HF dataset
    examples = df_to_hf_ds(
        examples,
        config=config,
        non_label_cols=["text"],
        text_column="text",
        problem_type=False,
    )

    # Predict on examples
    model_predictions = trainer.predict(examples)
    # Binarise predictions
    predictions = np.argmax(model_predictions.predictions, axis=-1)

    return predictions, model_predictions.metrics
