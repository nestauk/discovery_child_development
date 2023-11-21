# %%
from itertools import chain
import requests
from nesta_ds_utils.loading_saving import S3 as nesta_s3
from dotenv import load_dotenv
from typing import NoReturn, List, Any
import time

from discovery_child_development import S3_BUCKET, config

API_ROOT = "https://api.openalex.org/works?search=(child OR infant OR baby) AND "
S3_PATH = "metaflow"
SEED = config["seed"]
YEARS = config["openalex_years"]

load_dotenv()

# %%
KEYWORDS = ["technology", "monitor", "wearable", "development", "eye tracking"]

KEYWORDS = sorted(KEYWORDS)


# %%
def generate_queries(root=API_ROOT, keywords=KEYWORDS, years=YEARS):
    queries = []
    for k in keywords:
        for year in years:
            queries.append(f"{root}{k}&filter=publication_year:{year}")
    return queries


# %%
queries = generate_queries()

# %%
queries


# %%
def api_generator(query) -> list:
    """Generates a list of all URLs needed to completely collect
    all works relating to the list of concepts.

    Args:
        api_root : root URL of the OpenAlex API
        concept_ids : list of concept IDs to be queried

    Returns:
        all_pages: list of pages required to return all results
    """
    page_one = query
    print(f"Running API query {page_one}")
    total_results = requests.get(page_one).json()["meta"]["count"]
    print(f"Total number of hits: {total_results}")
    number_of_pages = -(total_results // -200)  # ceiling division
    all_pages = [
        f"{page_one}&per-page=200&cursor=" for _ in range(1, number_of_pages + 1)
    ]
    return all_pages


# %%
api_calls = [api_generator(q) for q in queries]
api_calls

# %%
api_calls_flat = list(chain.from_iterable(api_calls))
api_calls_flat[0:10]

# %%
# Get all results
outputs = []
cursor = "*"  # cursor iteration required to return >10k results
for call in api_calls_flat:
    query = f"{call}{cursor}"
    try:  # catch transient errors
        req = requests.get(query).json()
        print(f"Successfully accessed {query}")
        for result in req["results"]:
            outputs.append(result)
            cursor = req["meta"]["next_cursor"]
    except:
        print(f"Failure for query: {query}")
        pass

# %%
len(outputs)

# %%
