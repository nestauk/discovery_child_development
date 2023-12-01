# Openalex: Binary Classifier

In this folder you will find the code for the binary classifier used to identify relevant and irrelevant Openalex works.

### Instructions to replicate the pipeline

1. To pull in the openalex metadata for the classifier, run the following scripts in the repository root:

`python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True --random_sample True --chunk_size 1 --concept_ids openalex_broad_concepts --chunk_number 1`

This will pull in the metadata for Openalex works for the broad "not relevant" concepts. The metadata will be stored in the `openalex_broad_concepts` folder in the `metaflow` folder on S3.

For the relevant concepts, run the following script:
`python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True`

[IN PROGRESS] 2. Embedding the text data

3. Creating the training and validation set.
