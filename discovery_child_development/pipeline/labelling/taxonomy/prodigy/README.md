# Labelling data with GPT and Prodigy

This directory contains the code to label data with GPT and Prodigy. Currently this is all set up for multilabel classification, but it should be easy to adapt to other tasks.

## Setup

1. Create a new virtual environment and install `discovery_child_development/pipeline/labelling/taxonomy/prodigy/prodigy_requirements.txt`

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

## Other things to note

- You need to run things from the root level of the project so that you can import the various functions from the `discovery_child_development` package (i.e. do not `cd` into the prodigy directory)

- If you update the content of `discovery_child_development/pipeline/labelling/taxonomy/prompts/taxonomy_categories.json`, you should run `python discovery_child_development/pipeline/labelling/taxonomy/prodigy/format_labels.py` to regenerate the labels that the Prodigy app will use.
