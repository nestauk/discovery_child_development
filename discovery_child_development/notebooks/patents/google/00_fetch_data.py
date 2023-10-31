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
# Develop a sensible query to fetch relevant Google Patents queries (without incurring too much cost)

# +
from discovery_child_development.utils import bigquery, keywords as kw
from discovery_child_development import PROJECT_DIR

KEYWORD_FILE = PROJECT_DIR / "discovery_child_development/config/patents/keywords.txt"
PATENT_PATH = "data/patents/"
# -

keywords = kw.get_keywords(KEYWORD_FILE)
keywords = kw.deduplicate_keywords(
    keywords
    + kw.replace_word(keywords, "child", "infant")
    + kw.replace_word(keywords, "child", "baby")
)

query_keywords_path = KEYWORD_FILE.parent / "keywords_query.txt"
kw.save_keywords(keywords, query_keywords_path)

client = bigquery.create_client()

query = bigquery.create_patents_query(keywords)
print(query)

bigquery.dry_run(client, query)

query_df = client.query(query).to_dataframe()
len(query_df)

bigquery.upload_query_to_s3(
    query_name="GooglePatents",
    path=PATENT_PATH,
    query_df=query_df,
    query=query,
    metadata=[KEYWORD_FILE, query_keywords_path],
)
