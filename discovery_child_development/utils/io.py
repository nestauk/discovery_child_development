"""
Various input output utilities
"""
from discovery_child_development import PROJECT_DIR, logging
from yaml import safe_load
from typing import TypeVar, Iterator
import pathlib
import pickle
import json
import os
import pandas as pd
from zipfile import ZipFile

PathLike = TypeVar("PathLike", str, pathlib.Path, None)

DEF_CONFIG_PATH = PROJECT_DIR / "discovery_child_development/config"
VERBOSE = True


def import_config(filename: str, path=DEF_CONFIG_PATH) -> dict:
    """Load a config .yaml file"""
    with open(path / filename, "r", encoding="utf-8") as yaml_file:
        config_dict = safe_load(yaml_file)
    return config_dict
