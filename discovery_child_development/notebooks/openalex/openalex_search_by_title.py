# %% [markdown]
# This notebook was useful in the scoping phase of analysis when we were trying to figure out which OpenAlex concepts to use to retrieve relevant data for the project.
#
# So that the `pyalex` part of the code works, record your work email as `USER_EMAIL` in your .env file.
#
# The process is:
# * If you know of a work that may be a relevant target for this ISS phase, enter all or some of the title as the `PAPER_TITLE` variable below
# * Run the sections of code below to find the corresponding OpenAlex entry for that publication
# * Open `outputs/data/concepts.csv` for a tabular view of the concepts that the publication was tagged with
#
# See [this Google doc](https://docs.google.com/spreadsheets/d/1PObBjOHGiGJg-pfW70O6TBNl_MC-hrZMb3riMw3sChY/edit#gid=0)

# %%
import requests
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd

# %%
env_path = find_dotenv()
env_dir = os.path.dirname(env_path)
os.chdir(env_dir)
os.getcwd()

# %%
# This doesn't have to be the full title, but if you are not specific enough, you will get back too many entries to be helpful
PAPER_TITLE = "oligo-antigenic diet"

# %%
import pyalex
from pyalex import Works

pyalex.config.email = os.environ.get("USER_EMAIL")  # accessing the API politely

# %%
pager = Works().search_filter(title=PAPER_TITLE).paginate(per_page=200)

pages = []

for page in pager:
    pages.append(page)

len(pages)

# %%
len(pages[0])

# %%
for work in pages[0]:
    print(work["title"])

# %%
pages[0][2]["title"]

# %%
pages[0][0]["id"]

# %%
concepts_df = pd.DataFrame(pages[0][0]["concepts"])

concepts_df

# %%
concepts_df.to_csv("outputs/data/concepts.csv")

# %%
