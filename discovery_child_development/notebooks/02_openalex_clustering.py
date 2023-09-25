# %% [markdown]
# This notebook uses data created by `discovery_child_development/pipeline/openalex_metaflow.py`. The data produced by the metaflow pipeline gets saved to s3. For now, to run this notebook, you need to manually copy the data to `inputs/data/`.

# %%
import json
import os
from dotenv import find_dotenv
from time import time
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

import umap.umap_ as umap
import hdbscan
import altair as alt

from sklearn.feature_extraction.text import TfidfVectorizer

# We need to change wd in order to import utils functions
env_path = find_dotenv()
env_dir = os.path.dirname(env_path)
os.chdir(env_dir)
os.getcwd()

# Utils functions
from discovery_child_development.utils import openalex_utils

# %%
# Load raw data
input_data = os.path.join(
    "inputs",
    "data",
    "openalex-works_production-True_concept-C109260823_C2993937534_C2777082460_C2911196330_C2993037610_C2779415726_C2781192327_C15471489_C178229462_year-2023.json",
)

with open(input_data, "r") as f:
    openalex_data = json.load(f)

openalex_df = pd.DataFrame(openalex_data)

openalex_df.head()

# %% [markdown]
# Include only English works (previously, I left some texts in different languages in the datasets and they formed their own clusters that were pretty far away from everything else!)

# %%
openalex_en = openalex_df[openalex_df["language"] == "en"]
# Check how many works we lost by excluding other languages
print(
    f"Number of works that were not in English: {len(openalex_df) - len(openalex_en)}"
)

# %% [markdown]
# Check for and remove missing values in the 'title' and 'abstract' fields.

# %%
print(
    f"Number of NAs in 'abstract_inverted_index': {openalex_en['abstract_inverted_index'].isna().sum()}"
)
print(f"Number of NAs in 'title': {openalex_en['title'].isna().sum()}")

openalex_abstracts_en = openalex_en[openalex_en["abstract_inverted_index"].notnull()]
openalex_abstracts_en = openalex_abstracts_en[openalex_abstracts_en["title"].notnull()]

print(
    f"Number of NAs in 'abstract_inverted_index': {openalex_abstracts_en['abstract_inverted_index'].isna().sum()}"
)
print(f"Number of NAs in 'title': {openalex_abstracts_en['title'].isna().sum()}")

# %% [markdown]
# Deinvert the abstract and stick together the title and abstract. This mimics preprocessing done to create [this dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings).

# %%
openalex_abstracts_en.loc[:, "abstract"] = openalex_abstracts_en[
    "abstract_inverted_index"
].apply(lambda x: openalex_utils.deinvert_abstract(x))

openalex_abstracts_en.loc[:, "text"] = (
    openalex_abstracts_en["title"] + " " + openalex_abstracts_en["abstract"]
)

openalex_abstracts_en["text"].iloc[0:10]


# %% [markdown]
# # TFIDF

# %% [markdown]
# ## Define a spaCy tokenizer


# %%
def spacy_tokenizer(document):
    tokens = nlp(document)
    tokens = [
        token.lemma_
        for token in tokens
        if (
            token.is_stop == False
            and token.is_punct == False
            and token.lemma_.strip() != ""
        )
    ]
    return tokens


# %% [markdown]
# Define 3 different TFIDF vectorizers:
# * one using default settings
# * one using a spaCy tokenizer
# * one using a spaCy tokenizer and maximum and minimum document/term appearances

# %%
tfidf_vectorizer_plain = TfidfVectorizer(stop_words="english")

tfidf_vectorizer_spacy_tokenizer = TfidfVectorizer(
    input="content", tokenizer=spacy_tokenizer
)

tfidf_vectorizer_spacy_maxmin = TfidfVectorizer(
    input="content", tokenizer=spacy_tokenizer, max_df=0.5, min_df=5
)

# %%
# Turn dataframe column to list so that it is in the right format for the different vectorizers
openalex_text = openalex_abstracts_en["text"].tolist()


# %% [markdown]
# ## TFIDF with default settings


# %%
def apply_vectorizer(vectorizer, docs):
    t0 = time()
    vectors = vectorizer.fit_transform(docs)
    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {vectors.shape[0]}, n_features: {vectors.shape[1]}")
    return vectors, vectorizer


# %%
tfidf_vectors_plain, tfidf_vectorizer_plain = apply_vectorizer(
    tfidf_vectorizer_plain, openalex_text
)


# %%
def apply_umap_and_cluster(vectors, seed=None):
    if seed is not None:
        np.random.seed(seed)

    umap_vectors = umap.UMAP(
        n_neighbors=15, n_components=50, random_state=seed
    ).fit_transform(vectors)

    # use hdbscan to cluster, and assing all points to a cluster
    clusters = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    ).fit(umap_vectors)

    # use umap to reduce to 2-d
    umap_2d = umap.UMAP(
        n_neighbors=15, n_components=2, random_state=seed
    ).fit_transform(vectors)

    return clusters.labels_, umap_2d


# %%
tfidf_vectors_plain_labels, tfidf_vectors_plain_2d = apply_umap_and_cluster(
    tfidf_vectors_plain, seed=42
)

# %%
openalex_clusters = (
    openalex_abstracts_en[["title", "text"]]
    .assign(cluster_tfidf_plain=tfidf_vectors_plain_labels)
    .assign(x=tfidf_vectors_plain_2d[:, 0])
    .assign(y=tfidf_vectors_plain_2d[:, 1])
)

# %% [markdown]
# Around ~40% of observations are not assigned to a cluster!

# %%
openalex_clusters["cluster_tfidf_plain"].value_counts(normalize=True)


# %%
def plot_clusters(df, cluster_label, x_col, y_col):
    fig = (
        alt.Chart(df[df[cluster_label] != -1])
        .mark_circle()
        .encode(
            x=x_col,
            y=y_col,
            color=f"{cluster_label}:N",
            tooltip=["title", cluster_label],
        )
        .properties(width=800, height=600)
        .interactive()
    )

    return fig


# %%
fig1 = plot_clusters(openalex_clusters, "cluster_tfidf_plain", "x", "y")
fig1

# %%
from collections import defaultdict


def get_cluster_names(vectorizer, vectors, df, cluster_label_col):
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary to hold top words for each cluster
    top_words_per_cluster = defaultdict(list)

    # Number of top words you want to display per cluster
    n_top_words = 10

    # Iterate over each cluster and get top words
    for cluster_idx, tfidf_scores in enumerate(vectors):
        # Get indices of top n words within the cluster
        top_word_indices = tfidf_scores.toarray()[0].argsort()[: -n_top_words - 1 : -1]

        # Get the top words corresponding to the top indices
        top_words = [feature_names[i] for i in top_word_indices]

        # Append the words to the dictionary
        top_words_per_cluster[df.iloc[cluster_idx][cluster_label_col]] = top_words

    # Print the top words for each cluster
    cluster_names = [
        f"Cluster {cluster}: {', '.join(words)}"
        for cluster, words in top_words_per_cluster.items()
    ]

    return cluster_names


# %%
cluster_names = get_cluster_names(
    tfidf_vectorizer_plain,
    tfidf_vectors_plain,
    openalex_clusters,
    "cluster_tfidf_plain",
)
cluster_names

# %% [markdown]
# ## With spaCy tokenizer

# %%
# Takes much longer than using the inbuilt tokenizer
tfidf_vectors_spacy, tfidf_vectorizer_spacy = apply_vectorizer(
    tfidf_vectorizer_spacy_tokenizer, openalex_text
)

# %%
tfidf_vectors_spacy_labels, tfidf_vectors_spacy_2d = apply_umap_and_cluster(
    tfidf_vectors_spacy, seed=42
)

# %%
openalex_clusters = (
    openalex_clusters.assign(cluster_tfidf_spacy=tfidf_vectors_spacy_labels)
    .assign(x_spacy=tfidf_vectors_spacy_2d[:, 0])
    .assign(y_spacy=tfidf_vectors_spacy_2d[:, 1])
)

openalex_clusters.head()

# %% [markdown]
# Using the spacy tokenizer resulted in ~40-50% not assigned to a cluster... if anything, it got worse!

# %%
openalex_clusters["cluster_tfidf_spacy"].value_counts(normalize=True)

# %%
fig2 = plot_clusters(openalex_clusters, "cluster_tfidf_spacy", "x_spacy", "y_spacy")
fig2

# %%
tfidf_spacy_clusters = get_cluster_names(
    tfidf_vectorizer_spacy,
    tfidf_vectors_spacy,
    openalex_clusters,
    "cluster_tfidf_spacy",
)
tfidf_spacy_clusters

# %%
# Cluster 59 is about screening for ADHD
for x in openalex_clusters[openalex_clusters["cluster_tfidf_spacy"] == 59]["text"]:
    print(x)

# %%
# Cluster 28 is about technology in education
for x in openalex_clusters[openalex_clusters["cluster_tfidf_spacy"] == 28]["text"]:
    print(x)

# %%
# Cluster 48 is about games that have been developed to help children acquire skills
for x in openalex_clusters[openalex_clusters["cluster_tfidf_spacy"] == 48]["text"]:
    print(x)

# %%
# Cluster 14 is mostly about screen time
for x in openalex_clusters[openalex_clusters["cluster_tfidf_spacy"] == 14]["text"]:
    print(x)

# %% [markdown]
# ## TFIDF with a maximum and minimum document frequency

# %%
tfidf_vectors_spacy_maxmin, tfidf_vectorizer_spacy_maxmin = apply_vectorizer(
    tfidf_vectorizer_spacy_maxmin, openalex_text
)

# %%
(
    tfidf_vectors_spacy_maxmin_labels,
    tfidf_vectors_spacy_maxmin_2d,
) = apply_umap_and_cluster(tfidf_vectors_spacy_maxmin, seed=42)

# %%
openalex_clusters = (
    openalex_clusters.assign(
        cluster_tfidf_spacy_maxmin=tfidf_vectors_spacy_maxmin_labels
    )
    .assign(x_spacy_maxmin=tfidf_vectors_spacy_maxmin_2d[:, 0])
    .assign(y_spacy_maxmin=tfidf_vectors_spacy_maxmin_2d[:, 1])
)

openalex_clusters.head()

# %% [markdown]
# Around 40% assigned to the noise cluster again.

# %%
openalex_clusters["cluster_tfidf_spacy_maxmin"].value_counts(normalize=True)

# %%
fig3 = plot_clusters(
    openalex_clusters, "cluster_tfidf_spacy_maxmin", "x_spacy_maxmin", "y_spacy_maxmin"
)
fig3

# %%
tfidf_spacy_maxmin_clusters = get_cluster_names(
    tfidf_vectorizer_spacy_maxmin,
    tfidf_vectors_spacy_maxmin,
    openalex_clusters,
    "cluster_tfidf_spacy_maxmin",
)
tfidf_spacy_maxmin_clusters

# %%
# all about digital stuff!
for x in openalex_clusters[openalex_clusters["cluster_tfidf_spacy_maxmin"] == 48][
    "text"
]:
    print(x)

# %%

# %%

# %%

# %%

# %% [markdown]
# # Pre-existing embeddings

# %%
import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
t0 = time()

sentence_vectors_384 = model.encode(openalex_text, show_progress_bar=True)

print(f"vectorization done in {time() - t0:.3f} s")

# %%
minilm_labels, minimlm_vectors_2d = apply_umap_and_cluster(
    sentence_vectors_384, seed=42
)

# %%
openalex_clusters = (
    openalex_clusters.assign(cluster_minilm=minilm_labels)
    .assign(x_minilm=minimlm_vectors_2d[:, 0])
    .assign(y_minilm=minimlm_vectors_2d[:, 1])
)

openalex_clusters.head()

# %% [markdown]
# About 30% of observations are in the noise cluster, which seems like an improvement!

# %%
openalex_clusters["cluster_minilm"].value_counts(normalize=True)

# %%
fig4 = plot_clusters(openalex_clusters, "cluster_minilm", "x_minilm", "y_minilm")
fig4

# %%
cluster_df = (
    openalex_clusters.groupby("cluster_minilm").agg({"text": " ".join}).reset_index()
)

# %%
# Use TFIDF on the miniLM clusters
new_tfidf_vectors, new_tfidf_vectorizer = apply_vectorizer(
    tfidf_vectorizer_plain, cluster_df["text"].tolist()
)

minilm_plus_tfidf_cluster_names = get_cluster_names(
    new_tfidf_vectorizer, new_tfidf_vectors, cluster_df, "cluster_minilm"
)


# %%
minilm_plus_tfidf_cluster_names

# %%
print(cluster_df[cluster_df["cluster_minilm"] == 44]["text"])

# %% [markdown]
# ## Kmeans

# %%
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Instantiate the clustering model and visualizer
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2, 10))

visualizer.fit(sentence_vectors_384)  # Fit the data to the visualizer
visualizer.show()  # Finalize and render the figure

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 2, figsize=(15, 8))
for i in [2, 3, 4, 5]:
    """
    Create KMeans instances for different number of clusters
    """
    km = KMeans(
        n_clusters=i, init="k-means++", n_init=10, max_iter=100, random_state=42
    )
    q, mod = divmod(i, 2)
    """
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    """
    visualizer = SilhouetteVisualizer(km, colors="yellowbrick", ax=ax[q - 1][mod])
    visualizer.fit(sentence_vectors_384)

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortions = []
K = range(1, 40)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(sentence_vectors_384)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, "bx-")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("The Elbow Method showing the optimal k")
plt.show()

# %%

# %% [markdown]
# ## Spacy vectors
#
# Seeing as we already loaded a spaCy model, we might as well see how the embeddings perform!

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## FastText

# %%
openalex_text

# %%

# %%

# %%
import fasttext

fasttext_model = fasttext.train_unsupervised("openalex_text")

# %%
data["vec"] = data["processed_text"].apply(lambda x: model.get_sentence_vector(x))
