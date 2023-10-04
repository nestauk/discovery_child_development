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

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

load_dotenv()
S3_BUCKET = os.environ["S3_BUCKET"]
s3_client = boto3.client("s3")

# +
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

# +
MODEL = "distilbert-base-nli-stsb-mean-tokens"  # "all-MiniLM-L6-v2"

N_DIMENSIONS = 50

K = 40

SCORE_THRESHOLD = 0.5

SEED = 42

LEVEL_THRESHOLD = 1
# -

data_path = f"data/openAlex/concepts/concepts_metadata_{CONCEPTS}_year-2023.csv"
response = s3_client.get_object(Bucket=S3_BUCKET, Key=data_path)
csv_data = response["Body"].read()
concepts_df = pd.read_csv(BytesIO(csv_data), index_col=0)
concepts_df.head()

concepts_cleaned_df = concepts_df[concepts_df["score"] > SCORE_THRESHOLD]

# Filtering out non-relevant concepts makes quite a big difference
len(concepts_df) - len(concepts_cleaned_df)

# filter the granularity of concepts
concepts_cleaned_df = concepts_cleaned_df[
    concepts_cleaned_df["level"] > LEVEL_THRESHOLD
]

# # Unique concepts

concepts_df["display_name"].value_counts()

# # Cluster just the concepts

# +
import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL)
# -

unique_concepts = concepts_df["display_name"].unique().tolist()

t0 = time()
unique_concepts_384 = model.encode(unique_concepts, show_progress_bar=True)
print(f"vectorization done in {time() - t0:.3f} s")

# +
from sklearn.preprocessing import normalize

# Normalize the embeddings
normalized_embeddings = normalize(unique_concepts_384)
# -

umap_vectors = umap.UMAP(
    n_neighbors=15, n_components=N_DIMENSIONS, random_state=SEED
).fit_transform(normalized_embeddings)


# +
kmeans = KMeans(n_clusters=K, random_state=SEED)
kmeans_labels = kmeans.fit_predict(umap_vectors)

# Check the distribution of data points in each cluster
cluster_distribution = pd.Series(kmeans_labels).value_counts().sort_index()
cluster_distribution
# -

# Visualizing the distribution of data points in each cluster as a histogram
plt.figure(figsize=(10, 6))
cluster_distribution.plot(kind="bar", color="skyblue")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Data Points")
plt.title("Distribution of Data Points in Each Cluster")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

umap_2d = umap.UMAP(n_neighbors=15, n_components=2, random_state=SEED).fit_transform(
    normalized_embeddings
)

# +
unique_concepts_df = pd.DataFrame(concepts_df["display_name"].unique())
unique_concepts_df.columns = ["concept"]
unique_concepts_df = unique_concepts_df.assign(
    cluster=kmeans_labels, x=umap_2d[:, 0], y=umap_2d[:, 1]
)

unique_concepts_df["opacity"] = unique_concepts_df["cluster"].apply(
    lambda x: 0.3 if x == -1 else 1.0
)
unique_concepts_df["color"] = unique_concepts_df["cluster"].apply(
    lambda x: "noise" if x == -1 else x
)

# -

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

os.getcwd()

fig1.save(
    f"outputs/figures/unique_clusters_{K}_{MODEL}_{N_DIMENSIONS}_{SCORE_THRESHOLD}_{LEVEL_THRESHOLD}.html"
)
