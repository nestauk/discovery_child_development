"""
Works pipeline
--------------

A pipeline that takes a list of concept IDs and years, and outputs OpenAlex API results.

The thought behind this is to break the results into manageable yearly chunks. For a given year
and high level concept, the output works may be well over 2GB in size when saved to json.

Usage:

First, amend these variables:
* CONCEPT_IDS: list of OpenAlex concept IDs to be queried
* YEARS: list of years you want to retrieve publications from

To test the flow with just the first concept in the list:
python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production False

To fetch the full dataset:
python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True

If you want to run a random sample of works based on a concept and year, use the following command:
python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True --random_sample True

If you are reducing the chunk size, such that you will have more than 10 api calls per second (i.e the number of runs is higher than 10).
You will need to add --max-workers 10 to the command. Note: This will increase the time taken to run the flow significantly.

If you wish to change the concept IDs or years, you can do so by adding the following to the command line:
--concept_ids name_of_concepts_list_in_config --year_list name_of_years_list_in_config

To see a random sample of works regardless of concept, look at 00a_openalex_metaflow_random.py. Note: you will be able to get max 10000 works.
"""
import itertools
import requests
from metaflow import FlowSpec, S3, step, Parameter, retry, batch
from nesta_ds_utils.loading_saving import S3 as nesta_s3
from dotenv import load_dotenv
from typing import NoReturn, List, Any
import time

from discovery_child_development import S3_BUCKET, config

API_ROOT = "https://api.openalex.org/works?filter="
S3_PATH = "metaflow"
SEED = config["seed"]

load_dotenv()


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


def api_generator(api_root: str, concept_ids: List[str], random_sample: bool) -> list:
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
    print(f"Total number of pages queried: {number_of_pages}")
    if random_sample:
        all_pages = [
            f"{api_root}concepts.id:{concepts_text}&per-page=200&page={_}"
            for _ in range(1, number_of_pages + 1)
        ]
    else:
        all_pages = [
            f"{api_root}concepts.id:{concepts_text}&per-page=200"
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
    concepts = Parameter("concept_ids", default="openalex_concepts")
    years = Parameter("year_list", default="openalex_years")
    random_sample = Parameter("random_sample", default=False)
    number_works = Parameter(
        "number_works", default=10000
    )  # 10k is the max number of works per random sample
    chunk_size = Parameter(
        "chunk_size", default=40
    )  # 40 is the max number of concepts per query

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.concept_ids = config[self.concepts]
        self.year_list = config[self.years]
        self.next(self.generate_api_calls)

    @step
    def generate_api_calls(self):
        """Generates all API calls, if test, just one page"""
        # If production, generate all pages
        if self.production:
            concept_list = self.concept_ids
            year_list = self.year_list
        else:
            concept_list = self.concept_ids[:1]
            year_list = self.year_list[:1]
        # Generate chunks of concepts
        concept_chunks = get_chunks(concept_list, self.chunk_size)
        print(f"Number of concepts: {len(concept_chunks)}")
        # Get lists of queries for each chunk of concepts
        output_lists = []
        for chunk in concept_chunks:
            output_lists.append(generate_queries(chunk, year_list))
        # Flatten list of lists
        self.merged = list(itertools.chain.from_iterable(output_lists))
        print(f"Number of runs: {len(self.merged)}")
        if self.random_sample:
            self.merged = [
                api_call + "&sample=" + str(self.number_works) + f"&seed={SEED}"
                for api_call in self.merged
            ]
        self.next(self.retrieve_data, foreach="merged")

    @retry()
    @step
    def retrieve_data(self):
        """Returns all results of the API hits"""
        # Get list of API calls
        api_call_list = api_generator(API_ROOT, self.input, self.random_sample)
        # Get all results
        outputs = []
        cursor = "*"  # cursor iteration required to return >10k results
        for call in api_call_list:
            try:  # catch transient errors
                if self.random_sample:
                    req = requests.get(f"{call}").json()
                else:
                    req = requests.get(f"{call}&cursor={cursor}").json()
                for result in req["results"]:
                    outputs.append(result)
                cursor = req["meta"]["next_cursor"]
            except:
                pass
        print(self.input)
        # not ideal for multiple concepts, but works for now
        concept = self.input.split(",")[0]
        # Define a filename and save to S3
        year = self.input.split(":")[-1].replace("&", "_").replace("=", "-")
        filename = f"openalex-works_production-{self.production}_concept-{concept}_year-{year}.json"

        # Specify location to save the file within the bucket
        custom_path = f"{S3_PATH}/{self.concepts}/{filename}"

        nesta_s3.upload_obj(obj=outputs, bucket=S3_BUCKET, path_to=custom_path)

        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OpenAlexWorksFlow()
