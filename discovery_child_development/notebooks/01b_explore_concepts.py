# %% [markdown]
# This notebook explores the unique concepts that appear in the current extract of data. Note that the embedding and clustering is based purely on the unique concept names, not on the combinations of concepts that attach to particular abstracts.

# %%
import pandas as pd
import boto3
from io import BytesIO
import os
from dotenv import load_dotenv
from time import time
import umap.umap_ as umap
import hdbscan
import altair as alt
import numpy as np

# project-specific imports
from discovery_child_development.utils.cluster_analysis_utils import (
    hdbscan_param_grid_search,
    kmeans_param_grid_search,
    highest_silhouette_model_params,
)
import discovery_child_development.utils.cluster_analysis_utils as cau

# embeddings stuff
import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
# Set variables

SEED = 42

load_dotenv()
S3_BUCKET = os.environ["S3_BUCKET"]
s3_client = boto3.client("s3")

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

CONCEPTS = "|".join(CONCEPT_IDS)

YEARS = [2019, 2020, 2021, 2022, 2023]
years_list = [str(x) for x in YEARS]

# %%
# Load concepts metadata
data_path = f"data/openAlex/concepts/concepts_metadata_{CONCEPTS}_year-{'-'.join(years_list)}.csv"
response = s3_client.get_object(Bucket=S3_BUCKET, Key=data_path)
csv_data = response["Body"].read()
concepts_df = pd.read_csv(BytesIO(csv_data), index_col=0)
concepts_df.head()

# %%
# Include only concepts that have been scored with at least some relevance to the work (ie get rid of parent concepts)
concepts_cleaned_df = concepts_df[concepts_df["score"] > 0]

# %%
# Filtering out non-relevant concepts makes quite a big difference
len(concepts_df) - len(concepts_cleaned_df)

# %%
len(concepts_cleaned_df)

# %%
concepts_cleaned_df.head()

# %% [markdown]
# # Unique concepts

# %%
# Check out the most frequently occurring concepts
concepts_cleaned_df["display_name"].value_counts().head(20)

# %% [markdown]
# The top few are predictable e.g. Psychology, Early childhood. However, it is surprising to see "Computer science" in the mix as well! It is possible that the OpenAlex classifier has a bias towards computing and related concepts and is quite generous in tagging papers with these concepts.

# %% [markdown]
# # Cluster just the concepts

# %%
# At this point we create a column 'n_works' that may be useful later. If a concept only attached to 1 work in our corpus, do we still care about it?
unique_concepts_df = (
    concepts_cleaned_df.groupby("display_name").size().reset_index(name="n_works")
)
unique_concepts_df.head()

# %%
# Create embeddings for the unique concepts
t0 = time()
unique_concepts_384 = model.encode(
    unique_concepts_df["display_name"].tolist(), show_progress_bar=True
)
print(f"vectorization done in {time() - t0:.3f} s")

# %%
# Define HDBSCAN search parameters
hdbscan_search_params = {
    "min_cluster_size": [10, 20, 50],
    "min_samples": [1, 10, 15],
    "cluster_selection_method": ["leaf"],
    "prediction_data": [True],
}

umap_params = {
    "n_components": 50,  # apparently hdbscan does not work very well with more than 50 components
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

# %%
# reduce dimensionality of the embeddings
unique_concepts_50 = cau.umap_reducer(
    unique_concepts_384, umap_params, random_umap_state=SEED
)

# %%
# %%time
# Parameter grid search using HDBSCAN
hdbscan_search_results = hdbscan_param_grid_search(
    vectors=unique_concepts_50,
    search_params=hdbscan_search_params,
    have_noise_labels=False,  # we want everything to be forced into a cluster
)

# %%
hdbscan_search_results

# %%
# Find HDBSCAN model params with highest silhouette score
optimal_hdbscan_params = highest_silhouette_model_params(hdbscan_search_results)
optimal_hdbscan_params

# %% [markdown]
# We will check Kmeans results as well in case this method performs better than HDBSCAN.

# %%
# Define K-Means search parameters
kmeans_search_params = {
    "n_clusters": [5, 10, 15, 20, 25, 30, 35],
    "init": ["k-means++"],
}

# %%
# %%time
# Parameter grid search using K-Means
kmeans_search_results = kmeans_param_grid_search(
    vectors=unique_concepts_50, search_params=kmeans_search_params, random_seeds=[SEED]
)

# %%
# The silhouette scores for HDBSCAN turn out to be better
kmeans_search_results

# %%
# proceed with HDBSCAN clustering, seeing as that leads to a better silhouette score
hdbscan_labels = cau.hdbscan_clustering(
    unique_concepts_50, hdbscan_params=optimal_hdbscan_params, have_noise_labels=False
)

# %%
hdbscan_labels.head()

# %%
# Reduce original vectors to 2D for plotting
unique_concepts_2d = cau.reduce_to_2D(unique_concepts_384, random_state=SEED)

# %%
# Add 2D vectors into the dataframe for the sake of plotting. Add probability to the dataframe as well for later use
unique_concepts_df = unique_concepts_df.assign(
    cluster=hdbscan_labels["labels"],
    probability=hdbscan_labels["probability"],
    x_hdbscan=unique_concepts_2d[:, 0],
    y_hdbscan=unique_concepts_2d[:, 1],
)

# %%
fig_hdbscan = (
    alt.Chart(
        unique_concepts_df[unique_concepts_df["probability"] > 0.1]
    )  # filtering out some rows because altair has a limit on how many data points you can plot
    .mark_circle()
    .encode(
        x="x_hdbscan",
        y="y_hdbscan",
        color=alt.Color("cluster:N", legend=alt.Legend(title="cluster")),
        tooltip=["display_name", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig_hdbscan

# %% [markdown]
# # Check which concepts were assigned to which cluster
#
# In this section, we check the probability with which concepts were assigned to their clusters. It turns out there is a bimodal distribution of probabilities (probably because we forced everything to be assigned to a cluster instead of allowing a noise cluster?). We divide concepts into "core" and "peripheral", where "core" ones are ones that were assigned to their clusters with a high probability, and "peripheral" are the ones that were assigned with a low probability.

# %%
# The distribution of probabilities is bimodal
unique_concepts_df["probability"].hist()

# %%
# Separate out high probability and low probability concepts for each cluster
hdbscan_highest_prob_docs = unique_concepts_df[unique_concepts_df["probability"] > 0.5]
len(unique_concepts_df) - len(hdbscan_highest_prob_docs)

# %%
hdbscan_docs_grouped = (
    hdbscan_highest_prob_docs.groupby("cluster")
    .agg({"display_name": ";".join})
    .reset_index()
)

hdbscan_docs_grouped

# %%
hdbscan_low_prob_docs = unique_concepts_df[unique_concepts_df["probability"] <= 0.5]

hdbscan_low_prob_docs_grouped = (
    hdbscan_low_prob_docs.groupby("cluster")
    .agg({"display_name": ";".join})
    .reset_index()
)

hdbscan_low_prob_docs_grouped

# %%
hdbscan_docs_grouped = hdbscan_docs_grouped.rename(
    columns={"display_name": "high_prob_concepts"}
)
hdbscan_low_prob_docs_grouped = hdbscan_low_prob_docs_grouped.rename(
    columns={"display_name": "low_prob_concepts"}
)

# %%
hdbscan_docs_grouped = hdbscan_docs_grouped.merge(
    hdbscan_low_prob_docs_grouped, on="cluster", how="left"
)
hdbscan_docs_grouped.head()

# %%
hdbscan_docs_grouped.to_csv("hdbscan_highest_prob_concept_groups.csv")

# %% [markdown]
# # Most representative publication per concept
#
# It might be useful to know for each concept, which work got the highest score for that cluster.

# %%
most_relevant_work_per_concept = concepts_cleaned_df.loc[
    concepts_cleaned_df.groupby("display_name")["score"].idxmax()
][["concept_id", "display_name", "level", "openalex_id", "title", "score"]]

most_relevant_work_per_concept.head()

# %%
most_relevant_work_per_concept[
    most_relevant_work_per_concept["display_name"] == "Emerging technologies"
]

# %%
most_relevant_work_per_concept[
    most_relevant_work_per_concept["display_name"] == "Assistive technology"
]

# %%
unique_concepts_df["core_peripheral"] = np.where(
    unique_concepts_df["probability"] > 0.5, "core", "peripheral"
)

unique_concepts_df = unique_concepts_df.rename(
    columns={"probability": "cluster_probability"}
)

unique_concepts_df = unique_concepts_df[
    ["display_name", "cluster", "cluster_probability", "core_peripheral", "n_works"]
]

unique_concepts_df.head()

# %%
# This should be 0
len(most_relevant_work_per_concept) - len(unique_concepts_df)

# %%
concept_clusters_df = unique_concepts_df.merge(
    most_relevant_work_per_concept, on="display_name", how="left"
)

# %%
concept_clusters_df = concept_clusters_df.sort_values(
    by=["cluster", "cluster_probability"], ascending=False
)

concept_clusters_df.head(20)

# %%
concept_clusters_df.to_csv("concept_clusters.csv")

# %%
