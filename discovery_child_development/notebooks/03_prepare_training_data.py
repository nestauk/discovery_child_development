# %% [markdown]
# # Access the taxonomy from Google Sheets

# %%
import pandas as pd
from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import os
import re
import logging
from sklearn.preprocessing import MultiLabelBinarizer

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

from nesta_ds_utils.loading_saving import S3

# Download the credentials from s3 in case they are not already stored locally
from discovery_child_development.utils import bigquery

bigquery.find_credentials("GOOGLE_SHEETS_CREDENTIALS")


# %%
# Helpful functions


def extract_concept_substring(s):
    pattern = re.compile(r"\/(C\d+)$")
    match = pattern.search(s)
    return match.group(1) if match else None


# %%
# Define the scope for the Google Sheets API (we only want Google sheets)
scope = ["https://spreadsheets.google.com/feeds"]

# Provide the path to the downloaded service account key
credentials_file_path = f'../../{os.environ["GOOGLE_SHEETS_CREDENTIALS"]}'

# Authenticate using the credentials JSON file
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    credentials_file_path, scope
)

# The Google Sheets document ID (can be extracted from the URL)
spreadsheet_id = "1KIXSjS2MktbUKuyM-lprSK56kOh14u_gmHyGdHce87w"

# Optionally, if you have a specific range you want to read, you can provide the sheet name and range
# For example, 'Sheet1' or 'Sheet1!A1:C3'
wks_name = "initial_taxonomy"

# Load the data into a pandas DataFrame
data = g2d.download(
    spreadsheet_id, wks_name, credentials=credentials, col_names=True, row_names=True
)

# %%
data["Relevant?"].value_counts()

# %%
# Only retain concepts that have been determined to be relevant
taxonomy_df = data[data["Relevant?"].isin(["Y", "Y?"])].copy()

# check that there are no duplicate concepts
len(taxonomy_df) - len(taxonomy_df["display_name"].unique())

# %%
# check that the number of unique names and the number of unique ids are the same
len(taxonomy_df["display_name"].unique()) == len(taxonomy_df["concept_id"].unique())

# %%
# Create a list of concepts that can be used for filtering other dataframes

pattern = re.compile(r"\/(C\d+)$")

concept_ids = [
    pattern.search(string).group(1)
    for string in taxonomy_df["concept_id"].tolist()
    if pattern.search(string)
]

# %% [markdown]
# # Load abstracts

# %%
abstracts_filename = "openalex_abstracts_C109260823|C2993937534|C2777082460|C2911196330|C2993037610|C2779415726|C2781192327|C15471489|C178229462_year-2019-2020-2021-2022-2023.csv"

# %%
openalex_data = S3.download_obj(
    "discovery-iss",
    path_from=f"data/openAlex/{abstracts_filename}",
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

openalex_data.head()

# %% [markdown]
# # Load concepts

# %%
concepts_file = "concepts_metadata_C109260823|C2993937534|C2777082460|C2911196330|C2993037610|C2779415726|C2781192327|C15471489|C178229462_year-2019-2020-2021-2022-2023.csv"

openalex_concepts = S3.download_obj(
    "discovery-iss",
    path_from=f"data/openAlex/concepts/{concepts_file}",
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

openalex_concepts.head()

# %% [markdown]
# ## Check distribution of scores
#
# OpenAlex uses [a threshold of 0.3](https://docs.openalex.org/api-entities/works/work-object#concepts) to assign a concept to a work.

# %%
# Shows bimodal/multimodal distribution: there are a lot of very low scores close to 0, and another peak at around 0.5
openalex_concepts[["score"]].hist()


# %%
# Check the score distribution for each of the top-level concepts in the data to see if most observations in each bucket are above or below the 0.3 threshold


def hist_with_line(data, color):
    n, bins, patches = plt.hist(
        data, color=color, edgecolor="k"
    )  # Plotting the histogram
    plt.axvline(0.3, color="red", linestyle="--")  # Adding a vertical line at x=0.3


# Create a grid of histograms
g = sns.FacetGrid(
    openalex_concepts[openalex_concepts["level"] == 0],
    col="display_name",
    col_wrap=4,
    height=5,
)
g.map(hist_with_line, "score")

# Setting the same x-axis limits for all facets
x_min = 0  # define your own min value
x_max = 1  # define your own max value
g.set(xlim=(x_min, x_max))

plt.show()

# %% [markdown]
# Next steps:
# * Keep only concepts that are in the taxonomy
# * Keep only concepts with a score of >= 0.3
# * Use the filtered concepts metadata to filter the text data (left/right join)

# %%
# Subset the data using the 0.3 threshold
openalex_concepts_subset = openalex_concepts[openalex_concepts["score"] >= 0.3].copy()
logging.info(
    f"Prop of concepts tagged with a score less than 0.3: {(len(openalex_concepts) - len(openalex_concepts_subset))/len(openalex_concepts)}"
)
logging.info(f"N rows: {len(openalex_concepts_subset)}")

# we only need the concept substring, not the full url
openalex_concepts_subset["concept_substring"] = openalex_concepts_subset[
    "concept_id"
].apply(extract_concept_substring)

openalex_concepts_subset = openalex_concepts_subset[
    openalex_concepts_subset["concept_substring"].isin(concept_ids)
].copy()
logging.info(f"N rows: {len(openalex_concepts_subset)}")

openalex_concepts_subset.head()

# %%
openalex_concepts_subset.columns

# %%
openalex_data.columns

# %%
openalex_data = openalex_concepts_subset[
    ["openalex_id", "display_name", "concept_substring"]
].merge(openalex_data[["id", "text"]], left_on="openalex_id", right_on="id", how="left")

openalex_data = openalex_data.drop(columns=["id", "concept_substring"], axis=1)

openalex_data.head()

# %%
openalex_data = (
    openalex_data.groupby(["openalex_id", "text"])["display_name"]
    .agg(tuple)
    .reset_index()
)

# %%
openalex_data

# %%
mlb = MultiLabelBinarizer()

dummy_cols = pd.DataFrame(
    mlb.fit_transform(openalex_data["display_name"]),
    columns=mlb.classes_,
    index=openalex_data.index,
)

# %%
dummy_cols

# %%

# %%
openalex_data_wide = pd.pivot(
    openalex_data, index=["openalex_id", "text"], columns="display_name"
).reset_index()
openalex_data_wide.head()

# %%
