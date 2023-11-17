"""
Works pipeline
--------------

A pipeline that takes a list of years, and outputs a random sample of OpenAlex API results. 
You can specify the number of works you want to retrieve per year, with a maximum of 10,000.

Usage:

First, amend these variables:
* YEARS: list of years you want to retrieve publications from

To test the flow with just the first year in the list:
python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production False 

To fetch the full dataset:
python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True 
"""
import requests
from metaflow import FlowSpec, S3, step, Parameter, retry, batch
from nesta_ds_utils.loading_saving import S3 as nesta_s3
from dotenv import load_dotenv
from typing import NoReturn, List, Any
import random
from discovery_child_development import PROJECT_DIR, S3_BUCKET, config

API_ROOT = "https://api.openalex.org/works?filter="
S3_PATH = "metaflow"
load_dotenv()


def generate_random_queries(years: List[str], number_works: int) -> List[str]:
    """Generates a list of queries for a random list of works for the
    years required.

    Args:
        years : list of years to be queried

    Returns:
        query_list : list of all queries
    """
    return [f"{year}&sample={number_works}" for year in years]


def api_generator_random(api_root: str, filter_call: str) -> list:
    """Generates a list of all URLs needed to collect a random sample

    Args:
        api_root : root URL of the OpenAlex API

        concept_ids : list of concept IDs to be queried

    Returns:
        all_pages: list of pages required to return all results
    """
    # Set a maximum number of pages to return
    page_one = f"{api_root}publication_year:{filter_call}"
    print(f"Running API query {page_one}")

    total_results = requests.get(page_one).json()["meta"]["count"]
    print(f"Total number of hits: {total_results}")

    number_of_pages = -(total_results // -200)  # ceiling division
    print(f"Total number of pages queried: {number_of_pages}")
    all_pages = [
        f"{api_root}publication_year:{filter_call}&per-page=200&cursor="
        for _ in range(1, number_of_pages + 1)
    ]

    return all_pages


class OpenAlexWorksFlowRandom(FlowSpec):
    production = Parameter("production", help="Run in production?", default=False)
    years = Parameter("year_list", default="openalex_years")
    number_works = Parameter("number_works", default=10000)

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

        if self.production:
            year_list = self.year_list
        else:
            year_list = self.year_list[:1]

        output_lists = generate_random_queries(year_list, self.number_works)
        self.merged = output_lists

        print(len(self.merged))
        self.next(self.retrieve_data, foreach="merged")

    @retry()
    @step
    def retrieve_data(self):
        """Returns all results of the API hits"""
        # Get list of API calls
        api_call_list = api_generator_random(API_ROOT, self.input)

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
        print(self.input)
        # Define a filename and save to S3
        year = self.input.split("&")[0]
        sample_size = self.input.split("=")[-1]
        filename = f"openalex-works_production-{self.production}_year-{year}_samplesize-{sample_size}.json"

        # Specify location to save the file within the bucket
        custom_path = f"{S3_PATH}/random_sample/{filename}"

        nesta_s3.upload_obj(obj=outputs, bucket=S3_BUCKET, path_to=custom_path)

        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OpenAlexWorksFlowRandom()
