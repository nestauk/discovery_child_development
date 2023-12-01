import boto3
from dotenv import load_dotenv
from itertools import chain
import json
from metaflow import FlowSpec, S3, step, Parameter, retry, batch
from nesta_ds_utils.loading_saving import S3 as nesta_s3
import requests
from typing import List
import time
import datetime

from discovery_child_development import S3_BUCKET, config
from discovery_child_development.utils import openalex_utils

API_ROOT = config["openalex_keywords_api_root"]
S3_PATH = "metaflow/openalex_keyword_search"
YEARS = config["openalex_years"]
KEYWORDS = config["openalex_keywords"]

# Output path
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
query_name = f"openalex_keywords_{timestamp}"
OUT_PATH = f"{S3_PATH}/{query_name}"

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


def save_keywords_to_s3(
    keywords: List[str], path: str, timestamp: str, file_prefix: str
) -> None:
    """
    Save the KEYWORDS list to a .txt file and upload it to S3.

    Args:
        keywords (List[str]): List of keywords to save.
        path (str): S3 path to upload to.
        timestamp (str): Timestamp to create a unique filename.
    """
    if isinstance(keywords, list):
        keywords_str = "\n".join(keywords)
    else:
        keywords_str = keywords

    filename = f"{file_prefix}_{timestamp}.txt"
    custom_path = f"{path}/{filename}"

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=S3_BUCKET, Key=custom_path, Body=keywords_str.encode("utf-8")
    )


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

        self.outputs = outputs
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join all the outputs from the parallel steps"""
        all_outputs = []
        for input in inputs:
            all_outputs.extend(input.outputs)

        # Save all outputs to a single JSON file
        self.save_all_outputs_to_s3(all_outputs)
        self.next(self.end)

    def save_all_outputs_to_s3(self, all_outputs):
        """Save all outputs to a single JSON file in S3"""
        file_name = f"openalex_keywords_combined.json"
        out_path = f"{OUT_PATH}_production_{self.production}"
        custom_path = f"{out_path}/{file_name}"

        s3_client = boto3.client("s3")
        data = json.dumps(all_outputs).encode("utf-8")
        s3_client.put_object(Bucket=S3_BUCKET, Key=custom_path, Body=data)
        print("Saved data")

        if self.production == False:
            keywords_to_save = KEYWORDS[:1]
            apis_to_save = apis_to_save = openalex_utils.generate_keyword_queries(
                API_ROOT, keywords_to_save, YEARS[:1]
            )
        else:
            keywords_to_save = KEYWORDS
            apis_to_save = openalex_utils.generate_keyword_queries(
                API_ROOT, KEYWORDS, YEARS
            )

        save_keywords_to_s3(keywords_to_save, out_path, timestamp, "keywords")
        print("Saved keywords")
        save_keywords_to_s3(apis_to_save, out_path, timestamp, "api_calls")
        print("Saved API calls")

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OpenAlexFlow()
