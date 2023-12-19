import pandas as pd

from discovery_child_development.getters import taxonomy

# this function downloads and saves the data locally, and also loads it in memory
data = pd.DataFrame(taxonomy.get_labelling_sample())

print(data.head())
