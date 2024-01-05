from prodigy.components.db import connect

db = connect()
all_dataset_names = db.datasets

print(all_dataset_names)
