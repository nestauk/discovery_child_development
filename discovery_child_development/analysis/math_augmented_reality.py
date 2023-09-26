import os
import pandas as pd
import pyalex
from pyalex import Works

PAPER_TITLE = "Mathematics learning with augmented reality"

pyalex.config.email = os.environ.get("USER_EMAIL")  # accessing the API politely

pager = Works().search_filter(title=PAPER_TITLE).paginate(per_page=200)

pages = []

for page in pager:
    pages.append(page)

idx = 0
augmented_reality = []

for work in pages[0]:
    print(idx, work["title"])
    augmented_reality.append((work["id"], work["doi"], work["title"]))
    idx += 1

math_augmented_reality = pd.DataFrame(
    augmented_reality, columns=["openalex_id", "doi", "title"]
)

math_augmented_reality.to_csv("outputs/data/math_augmented_reality.csv", index=False)
