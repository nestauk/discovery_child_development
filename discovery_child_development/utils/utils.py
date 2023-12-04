import json
import os
from typing import Generator


def load_jsonl(path: str):
    """Load a jsonl file into a list of dicts."""
    with open(path, "r") as f:
        results = [json.loads(line) for line in f.readlines()]  # noqa: F841
    return results


def batch(lst: list, n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_directory_if_not_exists(dir_path: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
