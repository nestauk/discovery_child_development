import json
from discovery_child_development import PROJECT_DIR
import pandas as pd

path = (
    PROJECT_DIR
    / "discovery_child_development/notebooks/labelling/data/relevance/test_data.jsonl"
)

# read a jsonl file
with open(path, "r") as f:
    results = [json.loads(line) for line in f.readlines()]  # noqa: F841

results = pd.DataFrame(results).sort_values("id")

print(results)
