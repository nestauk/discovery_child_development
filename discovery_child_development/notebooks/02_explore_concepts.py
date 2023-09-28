# This notebook explores the unique concepts that appear in the current extract of data. Note that the embedding and clustering is based purely on the unique concept names, not on the combinations of concepts that attach to particular abstracts (that will be the next notebook.)

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

load_dotenv()
S3_BUCKET = os.environ["S3_BUCKET"]
s3_client = boto3.client("s3")

CONCEPTS = "C109260823|C2993937534|C2777082460|C2911196330|C2993037610|C2779415726|C2781192327|C15471489|C178229462"

data_path = f"inputs/data/openAlex/concepts/concepts_metadata_{CONCEPTS}_year-2023.csv"
response = s3_client.get_object(Bucket=S3_BUCKET, Key=data_path)
csv_data = response["Body"].read()
concepts_df = pd.read_csv(BytesIO(csv_data), index_col=0)
concepts_df.head()

concepts_cleaned_df = concepts_df[concepts_df["score"] > 0]

# Filtering out non-relevant concepts makes quite a big difference
len(concepts_df) - len(concepts_cleaned_df)

# # Unique concepts

concepts_df["display_name"].value_counts()

# # Cluster just the concepts

# +
import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
# -

unique_concepts = concepts_df["display_name"].unique().tolist()

t0 = time()
unique_concepts_384 = model.encode(unique_concepts, show_progress_bar=True)
print(f"vectorization done in {time() - t0:.3f} s")

umap_vectors = umap.UMAP(
    n_neighbors=15, n_components=50, random_state=42
).fit_transform(unique_concepts_384)

concept_clusters = hdbscan.HDBSCAN(
    min_cluster_size=10,  # min_samples=15,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
).fit(umap_vectors)


umap_2d = umap.UMAP(n_neighbors=15, n_components=2, random_state=42).fit_transform(
    unique_concepts_384
)

# +
unique_concepts_df = pd.DataFrame(concepts_df["display_name"].unique())
unique_concepts_df.columns = ["concept"]
unique_concepts_df = unique_concepts_df.assign(
    cluster=concept_clusters.labels_, x=umap_2d[:, 0], y=umap_2d[:, 1]
)

unique_concepts_df["opacity"] = unique_concepts_df["cluster"].apply(
    lambda x: 0.3 if x == -1 else 1.0
)
unique_concepts_df["color"] = unique_concepts_df["cluster"].apply(
    lambda x: "noise" if x == -1 else x
)

# -

# In the plot below:
# * cluster 28 covers computing-related things
# * cluster 20 is statistics! These concepts probably come up in any paper that has a vagiely quant angle
# * 5 is cohort and longitudinal studies (very relevant when studying children's development)
# * Artificial intelligence ends up in the noise cluster

# +
fig1 = (
    alt.Chart(unique_concepts_df)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("color:N", legend=alt.Legend(title="cluster")),
        opacity="opacity:Q",
        tooltip=["concept", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig1
# -

fig1.save("outputs/figures/02_unique_clusters.html")

# Artificial intelligence always gets assigned to the noise cluster
unique_concepts_df[unique_concepts_df["concept"] == "Artificial intelligence"]

# ... but its parent concept, computer science, gets assigned to cluster 8
unique_concepts_df[unique_concepts_df["concept"] == "Computer science"]

unique_concepts_df

# +
# Group the data by 'cluster_minilm' and aggregate the text for each cluster
grouped_text = (
    unique_concepts_df.groupby("cluster")
    .agg(
        {
            "concept": ", ".join,
            #'cluster': 'size'
        }
    )
    .reset_index()
)  # .rename(columns={'cluster': 'count'})
# openalex_clusters.groupby('cluster_minilm')['text'].apply(' '.join).reset_index()

grouped_text.head()

# +
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_df=0.5, stop_words="english")

# Fit and transform the aggregated text for each cluster
tfidf_matrix = vectorizer.fit_transform(grouped_text["concept"])

# Extract the top terms for each cluster
top_terms = {}
feature_names = vectorizer.get_feature_names_out()
for i, row in enumerate(tfidf_matrix.toarray()):
    top_terms[grouped_text.iloc[i]["cluster"]] = [
        feature_names[index] for index in row.argsort()[-10:][::-1]
    ]

top_terms_df = pd.DataFrame(top_terms).T.reset_index()
top_terms_df.columns = ["cluster"] + [f"term_{i+1}" for i in range(10)]
# -

top_terms_df

top_terms_df.to_csv("top_terms_per_concept_cluster.csv", index=False)

# Observations:
# * cluster 10 seems to be about assessment -> relevant to this ISS3 project

unique_concepts_df = unique_concepts_df.merge(
    concepts_df, left_on="concept", right_on="display_name", how="left"
)
unique_concepts_df

# assessment
unique_concepts_df[unique_concepts_df["cluster"] == 10][["concept", "title"]]

# optics and vision
unique_concepts_df[unique_concepts_df["cluster"] == 16][["concept", "title"]]

# computing
unique_concepts_df[unique_concepts_df["cluster"] == 28]["title"].tolist()

#
