import json

from utils import flatten_dictionary

INPATH = "../notebooks/labelling/prompts/taxonomy/taxonomy_categories.json"
OUTPATH = "labels.txt"

if __name__ == "__main__":
    with open(INPATH) as json_file:
        categories = json.load(json_file)

    categories_flat = flatten_dictionary(categories)

    with open(OUTPATH, "w") as fp:
        for item in list(categories_flat.keys()):
            fp.write("%s\n" % item)
        print("Done")
