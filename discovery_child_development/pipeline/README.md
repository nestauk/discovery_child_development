# Pipelines for data downloads and processing

## OpenAlex publication data

### Downloading from OpenAlex API

To download OpenAlex data run the following two commands.

Download papers tagged with relevant concepts (tags)

```bash
python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True
```

The outputs will be saved in s3 in metaflow/openalex_concepts. Note that the results from this run might be distributed across multiple folders with slightly different timestamps - this has to do with some steps failing and then being retried with later timestamps.

Download papers with the relevant keywords

```bash
python discovery_child_development/pipeline/openalex/openalex_keyword_search.py run --production True
```

The outputs from this run will be saved in metaflow/openalex_keyword_search folders

### Preprocessing OpenAlex data

The next step is some light preprocessing of the OpenAlex data. The following script takes data from the metaflow folder and outputs it into data/openAlex/openalex_works_concepts_YYYYMMDD_HHMMSS/ with the timestamp indicating the time of the run.

```bash
python discovery_child_development/pipeline/openalex/01_preprocess_openalex.py
```

## Patents data download

Patent data is donwloaded with the following query

```bash
python discovery_child_development/pipeline/patents/00_fetch_patents.py
```

## Inference

### Check for relevant papers

Run the following script to check for relevant papers in the OpenAlex and Patent data

```bash
python discovery_child_development/pipeline/models/binary_classifier/05_inference_OpenAlex_Patents.py
```
