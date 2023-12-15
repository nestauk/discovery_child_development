import pandas as pd
import tarfile
import os.path
from pathlib import Path


def ensure_path_exists(path):
    """Ensures that a path exists. If it doesn't, it creates it.

    Args:
        path (str): Path to be checked.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def replace_binary_labels(
    df: pd.DataFrame,
    label_col: str = "labels",
    replace_cat: list = ["relevant", "not relevant"],
):
    """Replaces the labels in the dataframe with 1 (relevant) and 0 (not relevant).

    Args:
        df (pd.DataFrame): Dataframe to replace labels in.
        label_col (str, optional): Column that contains the labels. Defaults to "labels".
        replace_cat (list, optional): Categories to replace (order relevant first). Defaults to ["relevant", "not relevant"].

    Returns:
        pd.DataFrame: Dataframe with replaced labels.
    """
    df[label_col] = df[label_col].replace({replace_cat[0]: 1, replace_cat[1]: 0})
    return df


def make_tarfile(output_filename, source_dir):
    """Creates a tarfile from a folder

    Args:
        output_filename (str): Name of the file to be saved
        source_dir (str): Path to the folder to be zipped

    Returns:
        None: Saves the tarfile locally
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def extract_tarfile(filename, path_dir):
    """Extracts a tarfile to a folder

    Args:
        filename (str): Name of the file to be saved
        path_dir (str): Path to the folder to be unzipped

    Returns:
        None: Saves the tarfile locally
    """
    with tarfile.open(f"{path_dir}{filename}", "r:gz") as tar:
        tar.extractall(path=path_dir)
