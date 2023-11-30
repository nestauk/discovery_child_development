import boto3
from dotenv import load_dotenv
from itertools import chain
import json
from metaflow import FlowSpec, S3, step, Parameter, retry, batch
from nesta_ds_utils.loading_saving import S3 as nesta_s3
import requests
from typing import List
import time

from discovery_child_development import S3_BUCKET, config
from discovery_child_development.utils import openalex_utils

API_ROOT = config["openalex_keywords_api_root"]
S3_PATH = "metaflow/openalex_keyword_search"
YEARS = config["openalex_years"]
KEYWORDS = config["openalex_keywords"]

load_dotenv()


def api_generator(query: str) -> List[str]:
    """Generates a list of all URLs needed to completely collect
    all works relating to the list of concepts.

    Because only a certain number of results can be returned as one page, you
    may need to access multiple pages in order to get all the hits for a
    single query. This function takes a single query as input and by calculating
    how many results that query would return, then dividing that by 200, it figures
    out how many pages you will need to query. The result is a list where every
    item is the same API query, but crucially the length is the number of pages
    needed. That means that in the metaflow below, this function can be used in
    cursor paging.

    Args:
        query : the API query. This was produced by `generate_queries()` above.

    Returns:
        all_pages: list of pages required to return all results
    """
    print(f"Running API query {query}")
    total_results = requests.get(query).json()["meta"]["count"]
    print(f"Total number of hits: {total_results}")
    number_of_pages = -(total_results // -200)  # ceiling division
    all_pages = [f"{query}&per-page=200&cursor=" for _ in range(1, number_of_pages + 1)]
    return all_pages


class OpenAlexFlow(FlowSpec):
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
            keyword_list = KEYWORDS
            year_list = YEARS
        else:
            keyword_list = KEYWORDS[:1]
            year_list = YEARS[:1]
        self.merged = openalex_utils.generate_keyword_queries(
            API_ROOT, keyword_list, year_list
        )
        print(len(self.merged))
        self.next(self.retrieve_data, foreach="merged")

    @retry()
    # @batch(cpu=2, memory=48000)
    @step
    def retrieve_data(self):
        """Returns all results of the API hits"""
        # Get list of API calls
        api_call_list = api_generator(self.input)
        # Get all results
        outputs = []
        cursor = "*"  # cursor iteration required to return >10k results
        for call in api_call_list:
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
            time.sleep(2)
        filename = f"openalex_keywords_{self.production}.json"

        # Define a filename and save to S3
        year = self.input.split(":")[-1]
        filename = f"openalex_keywords_{self.production}_year-{year}.json"

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
    OpenAlexFlow()