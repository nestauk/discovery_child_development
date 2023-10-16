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

# # Testing connection with BigQuery

# +
from discovery_child_development.utils import bigquery

client = bigquery.create_client()

# Define the SQL query
sql = """
< add your query here >
"""

# Execute the query
query_job = client.query(sql)

# Fetch the results
results = query_job.result()
# -

next(results)
