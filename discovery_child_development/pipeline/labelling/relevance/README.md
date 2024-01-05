# Labelling data for relevance

This pipeline allows us to iteratively label research papers and abstracts for relevance to the project.

The pipeline was designed to be iterative as we used `gpt-4-1106-preview` which at this time has tokens-per-day limit and hence not all of the data can be labelled in one go. It also allows for flexibility to easily add new datasets in the future.

## Usage

Data can be labelled by running the following command:

```bash
python discovery_child_development/pipeline/labelling/relevance/relevance_labels.py --dataset dataset-name
```

`dataset-name` can be one of the following: `openalex`, `openalex_broad` or `patents`.

The labelled data is saved on S3 as a jsonl file, and new labels are being appended to the existing file. The jsonl file has the following structure:

```json
{
  "prediction": "one of Relevant, Not-relevant, Not-specified",
  "id": "unique ID of the data point",
  "source": "dataset name",
  "model": "LLM used for labelling",
  "timestamp": "YYYYMMDDHHMMSS",
  "text": "the text that was used as input for labelling"
}
```

The output file name and other parameters such as the model and temperature are defined in the `config.yaml` file.

You can also specify some other parameters using the command line, which might be helpful, for example, when testing the pipeline:

```bash
python discovery_child_development/pipeline/labelling/relevance/relevance_labels.py --dataset openalex --num_samples 5 --model gpt-3.5-turbo-1106 --output_filename testing
```

So, if labelling the data from scratch, you would need to run the pipeline multiple times, for the desired number of samples and datasets. For example:

```bash
python discovery_child_development/pipeline/labelling/relevance/relevance_labels.py --dataset openalex --num_samples 1000 ; \
python discovery_child_development/pipeline/labelling/relevance/relevance_labels.py --dataset patents --num_samples 1000 ; \
python discovery_child_development/pipeline/labelling/relevance/relevance_labels.py --dataset openalex_broad --num_samples 500
```

## How does it work?

In the background, we're using OpenAI API function calling for the labelling task.

The prompts and category definitions are provided in the `prompts` folder:

- `categories.json` defines the categories used for labelling
- `examples.json` has a few examples included in the prompt to make this a few-shot classification. This increase the prompt token count quite a bit, but should result in better performance (as indicated in the initial testing)
- `function.json` defines the function that is used to call the API. Read more [here](https://platform.openai.com/docs/guides/function-calling) and check out [this blog](https://medium.com/discovery-at-nesta/how-to-use-gpt-4-and-openais-functions-for-text-classification-ad0957be9b25) about function calling.
- `prompt.json` defines the instructions for the LLM, including the definition of the task and some background info
