# %% [markdown]
# # Access the taxonomy from Google Sheets

# %%
import pandas as pd
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import os
import logging
from sklearn.model_selection import train_test_split

from nesta_ds_utils.loading_saving import S3

from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.utils import bigquery, labelling_utils
from discovery_child_development.utils.io import import_config

bigquery.find_credentials("GOOGLE_SHEETS_CREDENTIALS")

GOOGLE_SHEETS_CREDENTIALS = os.path.join(
    PROJECT_DIR, os.environ["GOOGLE_SHEETS_CREDENTIALS"]
)

S3_BUCKET = os.environ["S3_BUCKET"]

PARAMS = import_config("config.yaml")

CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])

OUT_PATH = "data/openAlex/processed/"

SEED = 42


# %%
# Helpful functions!
def clean_string(s):
    s = s.replace(
        "?", ""
    )  # Remove question marks (do this *before* removing trailing spaces)
    s = s.strip()  # Remove leading/trailing whitespace
    s = " ".join(s.split())  # Remove duplicate spaces
    s = s.lower()  # Convert to lowercase
    return s


def access_google_sheet(google_credentials_json, sheet_id, sheet_name):
    # Define the scope for the Google Sheets API (we only want Google sheets)
    scope = ["https://spreadsheets.google.com/feeds"]

    # Authenticate using the credentials JSON file
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        google_credentials_json, scope
    )

    # Load the data into a pandas DataFrame
    data = g2d.download(
        sheet_id, sheet_name, credentials=credentials, col_names=True, row_names=True
    )

    return data


# %%
taxonomy_data = access_google_sheet(
    GOOGLE_SHEETS_CREDENTIALS,
    sheet_id="1KIXSjS2MktbUKuyM-lprSK56kOh14u_gmHyGdHce87w",
    sheet_name="initial_taxonomy",
)

# %%
taxonomy_data["n_works"] = taxonomy_data["n_works"].astype("int64")

# %%
# keep only concepts/categories that have been marked relevant
taxonomy_df = taxonomy_data[taxonomy_data["Relevant?"].isin(["Y"])].copy()
# only concepts with 10 or more publications
taxonomy_df = taxonomy_df[taxonomy_df["n_works"] >= 10]

# clean the sub-category column
taxonomy_df["Sub category"] = taxonomy_df["Sub category"].apply(clean_string)

# %%
taxonomy_df["Relevant?"].value_counts()

# %%
taxonomy_df["Sub category"].unique()

# %%
# Get rid of unnecessary valeus of sub-category
taxonomy_df = taxonomy_df[~taxonomy_df["Sub category"].isin([""])]

sorted(taxonomy_df["Sub category"].unique())

# %%
# check that there are no duplicate concepts
len(taxonomy_df) - len(taxonomy_df["display_name"].unique())

# %%
# check that the number of unique names and the number of unique ids are the same
len(taxonomy_df["display_name"].unique()) == len(taxonomy_df["concept_id"].unique())

# %%
# Get the IDs of concepts that we will use to filter the OpenAlex data
taxonomy_concept_ids = taxonomy_df["concept_id"].unique()

# %% [markdown]
# # Load abstracts

# %%
abstracts_filename = (
    f"openalex_abstracts_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023.csv"
)

# %%
openalex_data = S3.download_obj(
    S3_BUCKET,
    path_from=f"data/openAlex/{abstracts_filename}",
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

openalex_data.head()

# %% [markdown]
# # Load concepts metadata

# %%
concepts_file = f"concepts_metadata_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023.csv"

openalex_concepts = S3.download_obj(
    "discovery-iss",
    path_from=f"data/openAlex/concepts/{concepts_file}",
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

openalex_concepts.head()

# %% [markdown]
# Next steps:
# * Keep only concepts that are in the taxonomy
# * Use the filtered concepts metadata to filter the text data (left/right join)

# %%
openalex_concepts_subset = openalex_concepts[
    openalex_concepts["concept_id"].isin(taxonomy_concept_ids)
].copy()
logging.info(f"N rows lost: {len(openalex_concepts)-len(openalex_concepts_subset)}")

openalex_concepts_subset.head()

# %%
# merge taxonomy
openalex_concepts_subset = pd.merge(
    openalex_concepts_subset,
    taxonomy_df[["Sub category", "concept_id"]],
    how="left",
    on="concept_id",
)
openalex_concepts_subset = openalex_concepts_subset.rename(
    columns={"Sub category": "sub_category"}
)
openalex_concepts_subset.head()

# %%
# Check whether any works have been lost because they were not tagged with any concepts from the taxonomy
len(openalex_concepts["openalex_id"].unique()) - len(
    openalex_concepts_subset["openalex_id"].unique()
)

# %%
# Merge the abstracts, concepts metadata and taxonomy sub-categories
openalex_data = openalex_concepts_subset[
    ["openalex_id", "concept_id", "sub_category", "display_name", "level", "score"]
].merge(openalex_data[["id", "text"]], left_on="openalex_id", right_on="id", how="left")

openalex_data = openalex_data.drop(columns=["id"], axis=1)

openalex_data.head()

# %%
# Split IDs into random train and test subsets
unique_ids = openalex_data["openalex_id"].unique()

train_ids, test_ids = train_test_split(unique_ids, test_size=0.1, random_state=SEED)

train_df = openalex_data[openalex_data["openalex_id"].isin(train_ids)]
test_df = openalex_data[openalex_data["openalex_id"].isin(test_ids)]

# %%
# write to s3
S3.upload_obj(
    train_df,
    S3_BUCKET,
    f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_train.csv",
)
S3.upload_obj(
    test_df,
    S3_BUCKET,
    f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_test.csv",
)
