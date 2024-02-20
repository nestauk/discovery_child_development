# GPT: Binary Classifier

In this folder you will find the code for the binary classifier used to identify relevant and irrelevant Openalex works.

### Instructions to replicate the pipeline

The labelled data for this pipeline was created using Chat GPT. The data can be retrieved using the following getter:
`discovery_child_development/getters/labels.py`

1. Embedding the text data for the simple classifier using the"all-MiniLM-L6-v2" model:

`python discovery_child_development/pipeline/models/binary_classifier/01_embed_gpt_labelled_data.py`

2. Creating the training, validation and test set. The training set will be used to train the classifier and the validation set will be used to evaluate the performance of the classifier. The test set will be used to evaluate the performance of the classifier after it has been trained and validated.

`python discovery_child_development/pipeline/models/binary_classifier/02_binary_classifier_training_data.py`

3. Embedding the text data for the DistilBERT classifier using the "distilbert-base-uncased" model:

`python discovery_child_development/pipeline/models/binary_classifier/03_embed_training_data_hugging_face.py --production True`

4. (a) Training the simple classifier:

`python discovery_child_development/pipeline/models/binary_classifier/04a_train_simple_classifiers.py`

4. (b) Training the DistilBERT classifier:

`python discovery_child_development/pipeline/models/binary_classifier/04b_train_distilbert_classifier.py --production True`

For evaluation of the classifier on the test set, see 
`discovery_child_development/pipeline/models/binary_classifier/evaluating_datasets/evaluate_test_data.py`

Due to the results, we chose to use the DistilBERT classifier for the pipeline. The simple classifier is still available in the repository for reference.

5. To run the classifier on a new dataset, use the following script:
`discovery_child_development/pipeline/models/binary_classifier/05_inference.py`

To use different datasets, follow the instructions in the config file:
`discovery_child_development/config/labelling.yaml`
