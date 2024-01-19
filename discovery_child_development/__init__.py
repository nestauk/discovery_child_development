"""discovery_child_development."""
import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml

# Bucket name
S3_BUCKET = "discovery-iss"


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Define log output locations
info_out = str(PROJECT_DIR / "info.log")
error_out = str(PROJECT_DIR / "errors.log")

# Read log config file
_log_config_path = Path(__file__).parent.resolve() / "config/logging.yaml"
_logging_config = get_yaml_config(_log_config_path)
if _logging_config:
    logging.config.dictConfig(_logging_config)

# Define module logger
logger = logging.getLogger(__name__)

# base/global config
_base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
base_config = get_yaml_config(_base_config_path)

_config_path = Path(__file__).parent.resolve() / "config/config.yaml"
config = get_yaml_config(_config_path)

# classifier yamls
_taxonomy_path = Path(__file__).parent.resolve() / "config/taxonomy_classifier.yaml"
taxonomy_config = get_yaml_config(_taxonomy_path)

_binary_path = Path(__file__).parent.resolve() / "config/binary_classifier.yaml"
binary_config = get_yaml_config(_binary_path)

_detection_management_path = (
    Path(__file__).parent.resolve() / "config/detection_management_classifier.yaml"
)
detection_management_config = get_yaml_config(_detection_management_path)
