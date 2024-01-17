"""
Script to fetch patents from Google Patents BigQuery database.

NB: Running this will incur costs (scanning 1TB ~ £6)
If you're changing the query, first experiment with dry_run() to
get an estimate of the query size, or use BigQuery's query explorer.

Usage:

python discovery_child_development/pipeline/patents/00_fetch_patents.py
"""
from discovery_child_development.utils import google_utils, keywords as kw
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
        + kw.replace_word(keywords, "child", "toddler")
    )
    # Save a reference copy on local/GitHub for the most up date query
    query_keywords_path = KEYWORD_FILE.parent / "keywords_query.txt"
    kw.save_keywords(keywords, query_keywords_path)
    query = google_utils.create_patents_query(keywords)
    # Run the query
    client = google_utils.create_client()
    google_utils.dry_run(client, query)
    query_df = client.query(query).to_dataframe()
    # Upload query results and log the query itself to S3
    # Note that we're also saving the keywords used in the query together with the data
    google_utils.upload_query_to_s3(
        query_name="GooglePatents",
        path=PATENT_PATH,
        query_df=query_df,
        query=query,
        metadata=[KEYWORD_FILE, query_keywords_path],
    )
