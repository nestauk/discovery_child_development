# Openalex: Binary Classifier

In this folder you will find the code for the binary classifier used to identify relevant and irrelevant Openalex works.

### Instructions to replicate the pipeline

1. To pull in the openalex metadata for the classifier, run the following scripts in the repository root:

`python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True --random_sample True --chunk_size 1 --concept_ids openalex_broad_concepts`

This will pull in the metadata for Openalex works for the broad "not relevant" concepts. The metadata will be stored in the `openalex_broad_concepts` folder in the `metaflow` folder on S3.

For the relevant concepts, run the following script:
`python discovery_child_development/pipeline/openalex/00_openalex_metaflow.py run --production True`

2. Preprocess the Openalex metadata to retain only English works, remove titles in the EY concept seed list and ensure works returned have abstracts/titles:

`python discovery_child_development/pipeline/binary_classifier/01_preprocess_openalex_broad.py`

3. (a) Embedding the text data for the simple classifier using the"all-MiniLM-L6-v2" model:

`python discovery_child_development/pipeline/binary_classifier/02_embed_openalex_docs_broad.py`

3. (b) Embedding the text data for the DistilBERT classifier using the "distilbert-base-uncased" model:

`python discovery_child_development/pipeline/openalex/binary_classifier/04_embed_training_data_hugging_face.py`

4. Creating the training, validation and test set. The training set will be used to train the classifier and the validation set will be used to evaluate the performance of the classifier. The test set will be used to evaluate the performance of the classifier after it has been trained and validated.

`python discovery_child_development/pipeline/binary_classifier/03_binary_classifier_training_data.py`

This script creates 3 training sets with different proportions of relevant and irrelevant works for testing balanced and unbalanced training sets.

5. Training the simple classifier:

`python discovery_child_development/pipeline/openalex/binary_classifier/05a_train_simple_classifiers.py`

6. Training the DistilBERT classifier:

`python discovery_child_development/pipeline/binary_classifier/05b_train_distilbert_classifier.py`
