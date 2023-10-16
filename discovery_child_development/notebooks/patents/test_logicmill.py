# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: discovery_child_development
#     language: python
#     name: python3
# ---

# # Testing Logic Mill database with a simple example

# +
import requests
import json
from os import environ
import dotenv

dotenv.load_dotenv()

# api token
token = environ.get("LOGIC_MILL_TOKEN")

# api endpoint
url = "https://api.logic-mill.net/api/v1/graphql/"

# build graphql query
query = """
query searchDocuments($index: String!, $keyword: String!, $searchFields: [String!]) {
  searchDocuments(index: $index, keyword: $keyword, searchFields: $searchFields) {
    id
    documentParts {
      title
      abstract
    }
    metadata {
      createdAt
      aliases
    }
    vector
    url
  }
}
"""


# build variables
variables = {
    "keyword": "child development",
    "index": "uspto_cos",
    "searchFields": ["documentParts.title", "documentParts.abstract"],
}

headers = {
    "content-type": "application/json",
    "Authorization": "Bearer " + token,
}


# send request
r = requests.post(url, headers=headers, json={"query": query, "variables": variables})

# data
data = r.json()["data"]["searchDocuments"]
# -

# Individual elements for each document
data[0].keys()

for d in data:
    print(d["documentParts"]["title"])

for d in data:
    print(d["documentParts"]["abstract"])

#
