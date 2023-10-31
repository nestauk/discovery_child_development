"""
Script to fetch patents from Google Patents BigQuery database.

NB: Running this will incur costs (scanning 1TB ~ £6)
If you're changing the query, first experiment with dry_run() to
get an estimate of the query size, or use BigQuery's query explorer.

Usage:

python discovery_child_development/pipeline/patents/fetch_patents.py
"""
from discovery_child_development.utils import bigquery, keywords as kw
from discovery_child_development import PROJECT_DIR

KEYWORD_FILE = PROJECT_DIR / "discovery_child_development/config/patents/keywords.txt"
PATENT_PATH = "data/patents/"

if __name__ == "__main__":
    keywords = kw.get_keywords(KEYWORD_FILE)
    # Extend the set of keywords
    keywords = kw.deduplicate_keywords(
        keywords
        + kw.replace_word(keywords, "child", "infant")
        + kw.replace_word(keywords, "child", "baby")
    )
    # Save the final list of keywords
    query_keywords_path = KEYWORD_FILE.parent / "keywords_query.txt"
    kw.save_keywords(keywords, query_keywords_path)
    query = bigquery.create_patents_query(keywords)
    # Run the query
    client = bigquery.create_client()
    bigquery.dry_run(client, query)
    query_df = client.query(query).to_dataframe()
    # Upload query results and log the query itself to S3
    bigquery.upload_query_to_s3(
        query_name="GooglePatents",
        path=PATENT_PATH,
        query_df=query_df,
        query=query,
        metadata=[KEYWORD_FILE, query_keywords_path],
    )
