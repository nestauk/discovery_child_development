import streamlit as st
import pandas as pd
from discovery_child_development.utils.openai_utils import client
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nesta_ds_utils.loading_saving import S3

import os
import dotenv

dotenv.load_dotenv()


def retrieve_similar_vectors(
    query: str, model, vectors, data, datasets, topics, n: int = 10
) -> pd.DataFrame:
    """Retrieve similar vectors to a query

    Args:
        query (str): Text query
        n (int, optional): Number of similar vectors to retrieve. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame of similar vectors
    """
    _vectors_df = (
        vectors.query("dataset in @datasets")
        .query("major_category in @topics")
        .drop_duplicates(subset=["id"])
    )
    vector_list = _vectors_df["miniLM_384_vector"].tolist()
    query_vect = model.encode([query], show_progress_bar=False)
    cosine_similarities = cosine_similarity(query_vect, vector_list)
    return (
        _vectors_df[["id"]]
        .assign(similarity=cosine_similarities[0])
        .sort_values("similarity", ascending=False)
        .iloc[0:n]
        .merge(data, on="id", how="left")
        .assign(similarity=lambda df: df["similarity"].round(2))
    )[
        [
            "_id",
            "url",
            "text",
            "topic",
            "major_category",
            "amount",
            "dataset",
            "similarity",
        ]
    ]


possible_topics = ["openalex", "patents", "crunchbase", "gtr"]


# Streamlit app
def main():
    # Load in variables in the session state, to persist during the usage of the app
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "model" not in st.session_state:
        st.session_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    if "data" not in st.session_state:
        # Load in data
        st.session_state["data"] = S3.download_obj(
            bucket=os.environ["S3_BUCKET"],
            path_from="data/assistant/full_data_final.csv",
            download_as="dataframe",
        ).fillna({"major_category": ","})
    if "vectors" not in st.session_state:
        # Load in vectors
        vectors_df = S3.download_obj(
            bucket=os.environ["S3_BUCKET"],
            path_from="data/assistant/full_vectors_final.parquet",
            download_as="dataframe",
        ).merge(
            st.session_state["data"][
                ["id", "topic", "major_category", "dataset", "amount"]
            ],
            on="id",
            how="left",
        )
        vectors_df_exploded = vectors_df.assign(
            major_category=lambda df: df.major_category.apply(
                lambda x: [x.strip() for x in x.split(",")]
            )
        ).explode("major_category")
        st.session_state["vectors"] = vectors_df_exploded
    if "topics" not in st.session_state:
        st.session_state["topics"] = [
            s for s in vectors_df_exploded["major_category"].unique() if len(s) > 1
        ]
    if "summary" not in st.session_state:
        st.session_state["summary"] = None
    if "query" not in st.session_state:
        st.session_state["query"] = None
    if "prompt" not in st.session_state:
        default_prompt = "You are an expert researcher. Generate summaries of the two or three main themes based on the documents most relevant to the user query. Write in a succinct and clear manner, using bullet points for each theme."
        st.session_state["prompt"] = default_prompt

    # Title of the app
    st.title("AI Soft Play assistant")

    # User input for the query
    query = st.text_input("Enter a query:")
    st.session_state["query"] = query

    # Selecting datasets for the search
    datasets = st.multiselect(
        "Select datasets", possible_topics, default=possible_topics
    )
    # Selected taxonomy topics
    selected_topics = st.multiselect(
        "Select topics", st.session_state["topics"], default=st.session_state["topics"]
    )
    # Slider for number of results
    n = st.slider("Number of results to retrieve", min_value=1, max_value=50, value=10)
    # Finding the top N most similar results
    if st.button("Search"):
        results = retrieve_similar_vectors(
            query,
            model=st.session_state["model"],
            vectors=st.session_state["vectors"],
            data=st.session_state["data"],
            datasets=datasets,
            topics=selected_topics,
            n=n,
        )
        st.session_state["results"] = results
    # Showing the output table
    if st.session_state["results"] is not None:
        st.data_editor(
            st.session_state["results"],
            column_config={
                "url": st.column_config.LinkColumn(
                    "URL",
                    help="Link to more information",
                ),
            },
            hide_index=True,
        )
    # Generating a summary
    st.session_state["prompt"] = st.text_area(
        "Prompt", value=st.session_state["prompt"]
    )

    if st.button("Generate output"):
        texts = (
            "ID: "
            + st.session_state["results"]["_id"]
            + " | TEXT: "
            + st.session_state["results"]["text"]
        )
        texts = "\n\n".join(texts)
        prompt = f"{st.session_state['prompt']}\n\n#Innovation texts\nUse the following documents: {texts}\n\n#Additional instructions\nFocus on the documents most relevant to the user query: {st.session_state['query']}. Reference the documents in your response using the ID field. #Here's an example: \n\n[Theme or innovation idea title]\n [Summary or the idea] \n(IDs: [ID of the document, ID of another document])"
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=messages,
            temperature=0.6,
            max_tokens=None,
        )
        st.session_state["summary"] = output.choices[0].message.content
    if st.session_state["summary"] is not None:
        st.write(st.session_state["summary"])


main()
