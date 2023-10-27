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

# # Querying Google Patents
#
# Develop a sensible query to fetch relevant Google Patents queries (without incurring too much costs)

# +
from discovery_child_development import PROJECT_DIR
import pandas as pd
from typing import List
from discovery_child_development.utils import bigquery
from nesta_ds_utils.loading_saving import S3
import os

KEYWORD_FILE = "keywords.txt"
OUT_PATH = "data/patents/"


# get keywords from a txt file
def get_keywords(file: str) -> List[str]:
    """Get keywords from a txt file"""
    with open(file) as f:
        keywords = f.readlines()
    keywords = [x.strip() for x in keywords]
    return keywords


def create_query(search_terms: List[str]):
    """Create a query to fetch data from BigQuery"""

    # Create a list of 'LIKE' conditions for each search term
    like_conditions = {}
    for field in ["title", "abstract"]:
        like_conditions[field] = [
            f'gpr.{field} LIKE "%{term}%"' for term in search_terms
        ]

    # Join conditions with 'OR'
    combined_conditions_title = " OR ".join(like_conditions["title"])
    combined_conditions_abstract = " OR ".join(like_conditions["abstract"])

    q = f"""
  WITH
  pubs as (
    SELECT DISTINCT
      pub.publication_number
    FROM `patents-public-data.patents.publications` pub
      INNER JOIN `patents-public-data.google_patents_research.publications` gpr ON
      pub.publication_number = gpr.publication_number
    WHERE
      ({combined_conditions_title})
      OR ({combined_conditions_abstract})
      AND pub.grant_date BETWEEN 20190101 AND 20231231
  )

  SELECT
    gpr.publication_number,
    url,
    pub.grant_date,
    title,
    title_translated,
    abstract,
    abstract_translated,
    top_terms,
    embedding_v1,
  FROM `patents-public-data.patents.publications` pub
    INNER JOIN `patents-public-data.google_patents_research.publications` gpr ON
    pub.publication_number = gpr.publication_number
  WHERE
    gpr.publication_number IN (SELECT publication_number FROM pubs)
  """

    return q


# -

# ## Set up

client = bigquery.create_client()

# Get keywords
search_terms_ey = get_keywords(KEYWORD_FILE)
search_terms_ey

# Check query
query = create_query(search_terms_ey)
print(query)

# ### Get the data

query_df = client.query(create_query(search_terms_ey)).to_dataframe()

# +
if len(query_df) == 0:
    raise ValueError("No results for your search terms. Retry with another term.")
else:
    print("Search complete. {} assets selected.".format(len(query_df)))

query_df.head()
# -

embedding_dict = dict(
    zip(query_df.publication_number.tolist(), query_df.embedding_v1.tolist())
)

pd.set_option("display.max_colwidth", None)
query_df[["title", "grant_date", "url", "abstract"]].head()

S3.upload_obj(
    query_df,
    os.environ["S3_BUCKET"],
    f"{OUT_PATH}GooglePatents_test_data.csv",
)
