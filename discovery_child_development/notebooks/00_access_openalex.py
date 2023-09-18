# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: discovery_child_development
#     language: python
#     name: discovery_child_development
# ---

import requests
import pandas as pd
import json
import os
from dotenv import find_dotenv

# Set WD so that we can import utils functions:

# +
env_path = find_dotenv()

env_dir = os.path.dirname(env_path)

os.chdir(env_dir)

os.getcwd()
# -

from discovery_child_development.utils import openalex_utils

# ## OpenAlex Concepts
#
# You can actually get a table of all the concepts [here](https://docs.google.com/spreadsheets/d/1LBFHjPt4rj_9r0t0TTAlT68NwOtNH8Z21lBMsJDMoZg/edit#gid=575855905). The code below allows you to get all the concepts as a dataframe.

all_concepts = requests.get("https://api.openalex.org/concepts").json()

# `all_concepts` has two fields, 'meta' and 'results'. 'meta' gives you info about how many possible concepts there are.

all_concepts["meta"]

# Find out the number of concepts we got back from our request

len(all_concepts["results"])

# Each item in 'results' is in `dict` format, with the following keys:

all_concepts["results"][0].keys()

# Here we write to file and read back using pandas, just because the concepts are a bit more readable in dataframe format.

# +
concepts_json_object = json.dumps(all_concepts["results"])

with open("inputs/data/concepts.json", "w") as outfile:
    outfile.write(concepts_json_object)
# -

concepts_df = pd.read_json("inputs/data/concepts.json")

# Because we didn't paginate, we only got the first page of results - so we only have information about 25 concepts!

len(concepts_df)

concepts_df

# ## Get some abstracts
#
# In the next section we use [pyalex](https://pypi.org/project/pyalex/) to access the API and get some potentially relevant abstracts for us to explore.

# +
import pyalex
from pyalex import Works

pyalex.config.email = os.environ.get("USER_EMAIL")

search_query = "novel technology child development"
# -

# You have use paging to get all the results. The PyAlex package has [cursor paging](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging#cursor-paging) built in. Expect this code to take around 30 seconds to run:

# +
pager = (
    Works().filter(publication_year=2023).search(search_query).paginate(per_page=200)
)

pages = []

for page in pager:
    pages.append(page)
# -

# Check how many pages we got:

len(pages)

# We asked for 200 hits per page, so this code should return 200:

len(pages[0])

# Inspect the contents of the 0th page:

pages[0]

# Inspect the available fields in the 10th item of the 0th page:

pages[0][10].keys()

# The OpenAlex API gives you abstracts in the format of [inverted index](https://docs.openalex.org/api-entities/works/work-object#abstract_inverted_index). We have a utils function to turn an inverted index into normal text.

# +

openalex_utils.uninvert_index(pages[0][10]["abstract_inverted_index"])
# -

# Currently, we have a list of pages, and each page is a list of works. We can unnest the outer list so that we just have a list of works.

pages_unnested = [item for sublist in pages for item in sublist]

# Check total number of works that we got back using our search query and filters:

len(pages_unnested)

# Save the file locally:

# +
test_data_json_object = json.dumps(pages_unnested)

with open("inputs/data/test_data.json", "w") as outfile:
    outfile.write(test_data_json_object)
# -
