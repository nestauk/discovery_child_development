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

# %%
import requests
import pandas as pd
import json
import os
from dotenv import find_dotenv
import s3fs

# %% [markdown]
# Set WD so that we can import utils functions:

# %%
env_path = find_dotenv()

env_dir = os.path.dirname(env_path)

os.chdir(env_dir)

os.getcwd()

# %%
from discovery_child_development.utils import openalex_utils

# %% [markdown]
# ## OpenAlex Concepts
#
# You can actually get a table of all the concepts [here](https://docs.google.com/spreadsheets/d/1LBFHjPt4rj_9r0t0TTAlT68NwOtNH8Z21lBMsJDMoZg/edit#gid=575855905). The code below allows you to get all the concepts as a dataframe.

# %%
all_concepts = requests.get("https://api.openalex.org/concepts").json()

# %% [markdown]
# `all_concepts` has two fields, 'meta' and 'results'. 'meta' gives you info about how many possible concepts there are.

# %%
all_concepts["meta"]

# %% [markdown]
# Find out the number of concepts we got back from our request

# %%
len(all_concepts["results"])

# %% [markdown]
# Each item in 'results' is in `dict` format, with the following keys:

# %%
all_concepts["results"][0].keys()

# %% [markdown]
# Here we write to file and read back using pandas, just because the concepts are a bit more readable in dataframe format.

# %%
concepts_json_object = json.dumps(all_concepts["results"])

with open("inputs/data/concepts.json", "w") as outfile:
    outfile.write(concepts_json_object)

# %%
concepts_df = pd.read_json("inputs/data/concepts.json")

# %% [markdown]
# Because we didn't paginate, we only got the first page of results - so we only have information about 25 concepts!

# %%
len(concepts_df)

# %%
concepts_df

# %% [markdown]
# ## Get abstracts using PyAlex and search functionality
#
# In the next section we use [pyalex](https://pypi.org/project/pyalex/) to access the API and get some potentially relevant abstracts for us to explore.

# %%
import pyalex
from pyalex import Works

pyalex.config.email = os.environ.get("USER_EMAIL")  # accessing the API politely

search_query = "novel technology child development"

# %% [markdown]
# You have use paging to get all the results. The PyAlex package has [cursor paging](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging#cursor-paging) built in. Expect this code to take around 30 seconds to run:

# %%
pager = (
    Works().filter(publication_year=2023).search(search_query).paginate(per_page=200)
)

pages = []

for page in pager:
    pages.append(page)

# %% [markdown]
# Check how many pages we got:

# %%
len(pages)

# %% [markdown]
# We asked for 200 hits per page, so this code should return 200:

# %%
len(pages[0])

# %% [markdown]
# The OpenAlex API gives you abstracts in the format of [inverted index](https://docs.openalex.org/api-entities/works/work-object#abstract_inverted_index). We have a utils function to turn an inverted index into normal text.

# %%

openalex_utils.uninvert_index(pages[0][10]["abstract_inverted_index"])

# %% [markdown]
# Currently, we have a list of pages, and each page is a list of works. We can unnest the outer list so that we just have a list of works.

# %%
pages_unnested = [item for sublist in pages for item in sublist]

# %% [markdown]
# Save the file locally:

# %%
filename = f'openalex_2023_{search_query.replace(" ", "_")}.json'
filename

# %%
test_data_json_object = json.dumps(pages_unnested)

with open(f"inputs/data/{filename}", "w") as outfile:
    outfile.write(test_data_json_object)

# %% [markdown]
# Save the file to the s3 bucket too:

# %%
aws_key = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
s3_path = os.environ["S3_BUCKET"]

fs = s3fs.S3FileSystem(key=aws_key, secret=aws_secret)

with fs.open(f"{s3_path}/discovery_child_development/{filename}", "w") as f:
    f.write(f"{json.dumps(pages_unnested)}\n")

# %% [markdown]
# ## Access all works under a given concept
#
# This bit relies a lot on [code from the Genomics AI project](https://github.com/nestauk/ai_genomics/blob/dev/ai_genomics/pipeline/openalex/works_pipeline.py).
#
# We will get all works from 2023 that fall under a specific set of concepts.

# %%
CONCEPT_IDS = [
    "C109260823",  # child development
    "C2993937534",  # childhood development
    "C2777082460",  # early childhood
    "C2911196330",  # child rearing
    "C2993037610",  # child care
    "C2779415726",  # child protection
    "C2781192327",  # child behavior checklist
    "C15471489",  # child psychotherapy
    "C178229462",  # early childhood education
]

YEARS = [2023]

API_ROOT = "https://api.openalex.org/works?filter="


# %%
def generate_queries(concepts, years):
    """Generates a list of queries for the list of concepts and
    years required.

    Args:
        concepts : list of concepts to be queried
        years : list of years to be queried

    Returns:
        query_list : list of all queries
    """
    concepts_joined = "|".join(concepts)
    return [f"{concepts_joined},publication_year:{year}" for year in years]


query = generate_queries(CONCEPT_IDS, YEARS)

# %%
# Find out how many results in total our query would get. We need this info for pagination
page_one = f"{API_ROOT}concepts.id:{query[0]}"
print(page_one)

total_results = requests.get(page_one).json()["meta"]["count"]

# %%
#  Get the URLs that we'll need to access all pages (really it's just the same url lots of times, and each time we'll change the cursor value)
number_of_pages = -(total_results // -200)  # ceiling division
all_pages = [
    f"{API_ROOT}concepts.id:{query[0]}&per-page=200&cursor="
    for _ in range(1, number_of_pages + 1)
]

# %%
# Get all results
outputs = []
cursor = "*"  # cursor iteration required to return >10k results
for call in all_pages:
    try:  # catch transient errors
        req = requests.get(f"{call}{cursor}").json()
        for result in req["results"]:
            outputs.append(result)
        cursor = req["meta"]["next_cursor"]
    except:
        pass

# %% [markdown]
# It is possible that works appear multiple times in the data if they are linked to multiple concepts eg both 'child development' and 'childhood development'. In practice this doesn't seem to be the case. When we run the code below, the deduplicated list of works turns out to be the same length as the non-deduped list.

# %%
# Deduplicate based on 'id'
deduplicated_data = {entry["id"]: entry for entry in outputs}.values()

# Convert back to a list
deduplicated_data = list(deduplicated_data)

# %%
# Calculate whether there are more works in the original list than in the deduplicated list
len(outputs) - len(deduplicated_data)

# %%
filename = "openalex_2023_concepts.json"

# %%
#  Save the output to the bucket
with fs.open(f"{s3_path}/discovery_child_development/{filename}", "w") as f:
    f.write(f"{json.dumps(pages_unnested)}\n")

# %%
