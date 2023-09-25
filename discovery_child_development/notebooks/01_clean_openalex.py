# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: discovery_child_development
#     language: python
#     name: discovery_child_development
# ---

# %% [markdown]
# ## Clean the OpenAlex abstracts
#
# Whereas the notebook `00_access_openalex` was about how to obtain data from OpenAlex, this notebook cleans the data and gets it into a format suitable for use with spaCy and Prodigy.

# %%
import json
import os
from dotenv import find_dotenv

# %%
env_path = find_dotenv()

env_dir = os.path.dirname(env_path)

os.chdir(env_dir)

os.getcwd()

# %%
from discovery_child_development.utils import openalex_utils

# %% [markdown]
# Read in the data:

# %%
with open("inputs/data/test_data.json", "r") as f:
    openalex_data = json.load(f)

# %% [markdown]
# Check the number of works in our data:

# %%
len(openalex_data)

# %% [markdown]
# Filter the works to just those that actually have abstracts (not all do!)

# %%
openalex_data_abstracts = [
    {
        "id": item["id"],
        "title": item["title"],
        "abstract_inverted_index": item["abstract_inverted_index"],
    }
    for item in openalex_data
    if "id" in item
    and "title" in item
    and "abstract_inverted_index" in item
    and item["abstract_inverted_index"] is not None
]

# %% [markdown]
# It turns out we lost ~400 works because they did not have abstracts:

# %%
len(openalex_data_abstracts)

# %% [markdown]
# Convert the abstracts from inverted indices to actual plain text. Save any errors for later inspection.

# %%
errors = []

for entry in openalex_data_abstracts:
    print(entry["id"])
    try:
        # calling it 'text' for compatibility with Prodigy
        entry["text"] = openalex_utils.uninvert_index(entry["abstract_inverted_index"])
    except Exception as e:
        print(f"This is the tricky one: {entry['id']}")
        errors.append(entry["id"])

# %% [markdown]
# Check how many errors we got:

# %%
errors

# %% [markdown]
# Only one abstract could not be converted to plain text. We can inspect this using the code below. We find that one word in the abstract contains square brackets, so maybe this is why we could not convert the abstract?

# %%
[
    item["abstract_inverted_index"]
    for item in openalex_data_abstracts
    if item["id"] == errors[0]
]

# %% [markdown]
# For Prodigy, the file needs to be saved as `.jsonl`

# %%
with open("inputs/data/openalex_data_abstracts.jsonl", "w") as jsonl_output:
    for entry in openalex_data_abstracts:
        json.dump(entry, jsonl_output)
        jsonl_output.write("\n")
