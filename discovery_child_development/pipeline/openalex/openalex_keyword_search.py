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
# YEARS = config["openalex_years"]
YEARS = list(range(2017, 2024))
KEYWORDS = config["openalex_keywords"]

TECH_KEYWORDS = '("chatgpt" OR "income" OR "early childhood education" OR "artificial intelligence" OR "assess" OR "assessment" OR "augmented reality" OR "autism" OR "behaviour" OR "behavior" OR "development" OR "eye tracking" OR "genetics" OR "income" OR "learning" OR "learning environment" OR "monitor" OR "psychotherapy" OR "randomised controlled trials" OR "robotics" OR "social media" OR "social services" OR "special need" OR "technology" OR "virtual reality" OR "wearable" OR "wearables" OR "app" OR "apps" OR "mobile" OR "math" OR "maths" OR "literacy" OR "reading" OR "read" OR "language" OR "communication" OR "machine learning" OR "deep learning" OR "generative ai" OR "speech")'
CHILD_KEYWORDS = '("child" OR "infant" OR "baby" OR "prenatal" OR "pregnancy" OR "toddler" OR "family" OR "parent" OR "pupil" OR "children" OR "babies" OR "toddlers" OR "families" OR "parents" OR "pupils")'
QUERY = f"https://api.openalex.org/works?search=(abstract:{CHILD_KEYWORDS} AND abstract:{TECH_KEYWORDS}) OR (title:{CHILD_KEYWORDS} AND title:{TECH_KEYWORDS})"
QUERIES = [f"{QUERY}&filter=publication_year:{year},type:article" for year in YEARS]


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
        # if self.production:
        #     keyword_list = KEYWORDS
        #     year_list = YEARS
        # else:
        #     keyword_list = KEYWORDS[:1]
        #     year_list = YEARS[:1]
        # self.merged = openalex_utils.generate_keyword_queries(
        #     API_ROOT, keyword_list, year_list
        # )
        self.merged = QUERIES
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
        for i, input in enumerate(inputs):
            # all_outputs.extend(input.outputs)
            # save the input list locally as a json file
            data = json.dumps(input.outputs).encode("utf-8")
            with open(f"output_{i}.json", "wb") as f:
                f.write(data)

        # Save all outputs to a single JSON file
        # self.save_all_outputs_to_s3(all_outputs)
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
            # apis_to_save = openalex_utils.generate_keyword_queries(
            #     API_ROOT, KEYWORDS, YEARS
            # )
            apis_to_save = QUERIES

        openalex_utils.save_keywords_to_s3(
            keywords_to_save, out_path, timestamp, "keywords"
        )
        print("Saved keywords")
        openalex_utils.save_keywords_to_s3(
            apis_to_save, out_path, timestamp, "api_calls"
        )
        print("Saved API calls")

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OpenAlexFlow()
