"""
works pipeline
--------------

A pipeline that takes a list of concept IDs and years, and outputs OpenAlex API results.

The thought behind this is to break the results into manageable yearly chunks. For a given year
and high level concept, the output works may be well over 2GB in size when saved to json.

Usage:

First, amend these variables:
* CONCEPT_IDS: list of OpenAlex concept IDs to be queried
* YEARS: list of years you want to retrieve publications from

To test the flow with just the first concept in the list:
python discovery_child_development/pipeline/openalex_metaflow.py run --production False

To fetch the full dataset:
python discovery_child_development/pipeline/openalex_metaflow.py run --production True
"""
import itertools
import json
import requests
from metaflow import FlowSpec, S3, step, Parameter, retry, batch
import os
import boto3
from dotenv import load_dotenv
from typing import NoReturn, List, Any

# Amend this to your desired concepts/years. OpenAlex allows up to 50 parameters
# per query, so code is included by default to chunk up the concepts into 40s.
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
    # "C138496976",  # developmental psychology (level 1).
]

YEARS = [2023]

API_ROOT = "https://api.openalex.org/works?filter="

load_dotenv()
# Define location to save the file
S3_BUCKET = os.environ["S3_BUCKET"]
# subfolder within the bucket
S3_PATH = "metaflow"


def generate_queries(concepts: List[str], years: List[str]) -> List[str]:
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


def api_generator(api_root: str, concept_ids: List[str]) -> list:
    """Generates a list of all URLs needed to completely collect
    all works relating to the list of concepts.

    Args:
        api_root : root URL of the OpenAlex API
        concept_ids : list of concept IDs to be queried

    Returns:
        all_pages: list of pages required to return all results
    """
    concepts_text = concept_ids
    page_one = f"{api_root}concepts.id:{concepts_text}"
    print(f"Running API query {page_one}")
    total_results = requests.get(page_one).json()["meta"]["count"]
    print(f"Total number of hits: {total_results}")
    number_of_pages = -(total_results // -200)  # ceiling division
    all_pages = [
        f"{api_root}concepts.id:{concepts_text}&per-page=200&cursor="
        for _ in range(1, number_of_pages + 1)
    ]
    return all_pages


def get_chunks(_list: List[Any], chunksize: int) -> List[List[Any]]:
    """
    Chunks a list.
    """
    chunks = [_list[x : x + chunksize] for x in range(0, len(_list), chunksize)]
    return chunks


class OpenAlexWorksFlow(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.generate_api_calls)

    @step
    def generate_api_calls(self):
        """Generates all API calls, if test, just one page"""
        # If production, generate all pages
        if self.production:
            concept_list = CONCEPT_IDS
            year_list = YEARS
        else:
            concept_list = CONCEPT_IDS[:1]
            year_list = YEARS[:1]
        # Generate chunks of concepts
        concept_chunks = get_chunks(
            concept_list, 40
        )  # 40 is the max number of concepts per query
        # Get lists of queries for each chunk of concepts
        output_lists = []
        for chunk in concept_chunks:
            output_lists.append(generate_queries(chunk, year_list))
        # Flatten list of lists
        self.merged = list(itertools.chain.from_iterable(output_lists))
        print(len(self.merged))
        self.next(self.retrieve_data, foreach="merged")

    @retry()
    @step
    def retrieve_data(self):
        """Returns all results of the API hits"""
        # Get list of API calls
        api_call_list = api_generator(API_ROOT, self.input)
        # Get all results
        outputs = []
        cursor = "*"  # cursor iteration required to return >10k results
        for call in api_call_list:
            try:  # catch transient errors
                req = requests.get(f"{call}{cursor}").json()
                for result in req["results"]:
                    outputs.append(result)
                cursor = req["meta"]["next_cursor"]
            except:
                pass
        # Define a filename and save to S3
        year = self.input.split(":")[
            -1
        ]  # not ideal for multiple concepts, but works for now
        concept = self.input.split(",")[0]
        filename = f"openalex-works_production-{self.production}_concept-{concept}_year-{year}.json"

        # Specify location to save the file within the bucket
        custom_path = f"{S3_PATH}/{filename}"

        # Use boto3 to save to the desired bucket
        s3_client = boto3.client("s3")
        data = json.dumps(outputs).encode("utf-8")  # Convert string to bytes
        s3_client.put_object(Bucket=S3_BUCKET, Key=custom_path, Body=data)

        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OpenAlexWorksFlow()
