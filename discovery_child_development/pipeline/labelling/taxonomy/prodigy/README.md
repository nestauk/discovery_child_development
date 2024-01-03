# Labelling data with GPT and Prodigy

This directory contains the code to label data with GPT and Prodigy. Currently this is all set up for multilabel classification, but it should be easy to adapt to other tasks.

## Setup

1. Create a new virtual environment with Python 3.10 (`conda create --name discovery_prodigy python=3.10.13`) and install `discovery_child_development/pipeline/labelling/taxonomy/prodigy/prodigy_requirements.txt`. Install pip in the conda env as well. Run `pip install -e .` so that all the local package functions work.

2. In the virtual environment, install Prodigy using a key - contact someone on the team for the key.

3. Make sure you have a `.env` file at the root of the project containing your OpenAI API key as `OPENAI_API_KEY`.

## Using the Prodigy app

### Run a test

To launch the app and label a couple of test texts, run

```
prodigy oa_classification test_data discovery_child_development/notebooks/labelling/prodigy/test_sample.jsonl -F discovery_child_development/notebooks/labelling/prodigy/taxonomy_classifier_recipe.py
```

You can export the data for inspection using:

```
prodigy db-out test_data > inputs/data/labelling/taxonomy/output/test_data_LABELLED.jsonl
```

### Label data for real

To launch the app and label the real data, first run this script to download the sample for labelling:

```
python discovery_child_development/pipeline/labelling/taxonomy/prodigy/get_data_sample.py
```

Then run the app:

```
prodigy oa_classification taxonomy_data inputs/data/labelling/taxonomy/input/training_validation_data_patents_openalex.jsonl -F discovery_child_development/pipeline/labelling/taxonomy/prodigy/taxonomy_classifier_recipe.py
```

To export the data locally, run:

```
prodigy db-out taxonomy_data > inputs/data/labelling/taxonomy/output/training_validation_data_patents_openalex_LABELLED.jsonl
```

Alternatively, to upload your data to S3, run:

```
python discovery_child_development/pipeline/labelling/taxonomy/prodigy/export_data_to_s3.py
```

(This script checks the data you have labelled locally against data already saved in S3 and uploads a deduplicated version of the data to S3.)

## Other things to note

- You need to run things from the root level of the project so that you can import the various functions from the `discovery_child_development` package (i.e. do not `cd` into the prodigy directory)

- If you update the content of `discovery_child_development/pipeline/labelling/taxonomy/prompts/taxonomy_categories.json`, you should run `python discovery_child_development/pipeline/labelling/taxonomy/prodigy/format_labels.py` to regenerate the labels that the Prodigy app will use.

- The `.jsonl` files may get saved with blank lines at the end, and then the Prodigy app will refuse to read them... Until we have properly fixed this, manually delete the empty lines at the end of the file.
