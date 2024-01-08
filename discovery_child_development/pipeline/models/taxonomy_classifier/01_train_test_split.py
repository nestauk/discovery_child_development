from discovery_child_development import logging, config, S3_BUCKET
from discovery_child_development.getters import taxonomy

from nesta_ds_utils.loading_saving import S3 as nesta_s3

import numpy as np
import pandas as pd
import random
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

SEED = config["seed"]
# There is not a way to set random seed with IterativeStratification, so we set a seed globally.
random.seed(SEED)
np.random.seed(SEED)

# For now we're splitting 70% train, 15% validation, and 15% hold-out test set.
TRAIN_PROP = 0.7
TEST_PROP = 0.15
VAL_PROP = 0.15

S3_PATH = "data/taxonomy_classifier/"


def stratified_split(df, col="labels", val=VAL_PROP, test=TEST_PROP, train=TRAIN_PROP):
    # One-hot encode the 'labels' column
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df[col])

    # Stratify and split the dataset into train and test/val by *labels*
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[(val + test), train]
    )
    train_indexes, test_val_indexes = next(stratifier.split(df, labels))

    # Create train set
    train_df = df.iloc[train_indexes]

    # Split the remaining data into test and validation sets
    test_val_df = df.iloc[test_val_indexes]
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[(val / (val + test)), (test / (val + test))],
    )
    test_indexes, validation_indexes = next(
        stratifier.split(test_val_df, labels[test_val_indexes])
    )

    test_df = test_val_df.iloc[test_indexes]
    validation_df = test_val_df.iloc[validation_indexes]

    return train_df, test_df, validation_df


if __name__ == "__main__":
    labelled_data = taxonomy.get_gpt_labelled_sample()

    logging.info(labelled_data["source"].value_counts())

    # Split into train, test and validation sets.
    # These should be stratified by (a) source and (b) labels.

    # Step 1: create 50/50 split where each split is stratified by source.
    # At the next step, we'll stratify just by labels, so it helps if we're doing that on a dataset
    # that is already evenly distributed by 'source'.
    # Splitting it 50/50 is arbitrary - the important thing is that you stratify by source, and then as
    # a separate step, stratify by labels (so I think it would also be fine if in this next bit of code,
    # you set test_size to 0.7, 0.4 or whatever value you want).
    df_a, df_b = train_test_split(
        labelled_data,
        test_size=0.5,
        stratify=labelled_data[["source"]],
        random_state=SEED,
    )

    # Within each half of the data, now create a train/test/val split stratified by labels.
    train_df_a, test_df_a, validation_df_a = stratified_split(df_a)
    train_df_b, test_df_b, validation_df_b = stratified_split(df_b)

    train_df = pd.concat([train_df_a, train_df_b], axis=0)
    logging.info(train_df["source"].value_counts(normalize=True))

    test_df = pd.concat([test_df_a, test_df_b], axis=0)
    logging.info(test_df["source"].value_counts(normalize=True))

    validation_df = pd.concat([validation_df_a, validation_df_b], axis=0)
    logging.info(validation_df["source"].value_counts(normalize=True))

    logging.info(f"Train df proportion:{len(train_df) / len(labelled_data)}")

    logging.info(f"Test df proportion:{len(test_df) / len(labelled_data)}")

    logging.info(f"Validation df proportion:{len(validation_df) / len(labelled_data)}")

    # write to s3
    logging.info("Uploading to S3...")
    datasets = {"train": train_df, "val": validation_df, "test": test_df}
    for key, value in datasets.items():
        nesta_s3.upload_obj(
            value,
            S3_BUCKET,
            f"{S3_PATH}taxonomy_labelled_data_{key}.parquet",
        )
    logging.info("Complete!")
