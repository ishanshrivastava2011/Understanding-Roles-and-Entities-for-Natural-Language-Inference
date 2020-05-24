# Enhancing natural language inference using new and expanded training data sets and new learning models

This repository contains the code and link to the datasets created and used in this work for all the experiments mentioned in the paper. 

## [Link to the Dataset and Model](https://drive.google.com/drive/folders/16gVgY_69luIv5JTvBbWKbGpKpr6uZjkA?usp=sharing)

This is the Google Drive link for the best performing model and all the datasets that were used for conducting all the experiments discussed in the paper including the two new datasets that we created.

## Requirements

- Python 3
> It's best to create a virtual environment
```
conda create -n allennlp_editable python=3.7
```

- Allen NLP 
> Install [Allennlp](https://github.com/allenai/allennlp#installing-from-source) from source. Follow the steps mentioned in this [link](https://github.com/allenai/allennlp#installing-from-source) to install allennlp from source. 

### (Optional) To create NER features for a new dataset
- Spacy
- Stanford Core NLP (stanford-ner-2018-10-16: download its zip file from https://drive.google.com/file/d/1q8hy3ZlxnURla0fFI8Ic9xc0Z05HxVbC/view?usp=sharing )

## Running the experiments

### Training Configs

These are the configuration files as required by the AllenNLP framework. All the configurations used in the work are saved in */new_experiments_iter2/training_config/* folder. The naming convention used for these files in this work is based on the kind of experiment. For example, the tranining config for the ESIM Lambda model trained on SNLI, NC (NER Changed) and RS (Role-Switched) datasets is named as *esim_lambda_snli_nc_rs.jsonnet*.

To perform any of the experiments mentioned in the paper, please download the datasets from the link provided and change the *train_data_path*, *validation_data_path* and *test_data_path* in the appropriate config file.

### Batch Script 

A sample batch script named *sample_batch_script.sh* shows how to run an experiment. This script contains AllenNLP's train, evaluate and predict commands. More details on these commands can be found in AllenNLP's [tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/walk_through_allennlp/training_and_evaluating.md)

To run different experiments, you will need to change three things in this script:
- *config_path* : This variable holds the path to the configuration files
- *serialization_dir* : This variable holds the path to the serialization directory where you would like to save the model and all the results.
- *test_data_path* : This variable holds the path to the directory that contains all the test data.

### Output for the experiments

Based on the sample script given, you should expect the following files as the output:
- The train command results in model.tar.gz file as the serialized model.
- The evaluate command results in *testEvaluate_snli.txt*, *testEvaluate_NC.txt* and *testEvaluate_RS.txt* files that contain the loss and accuracies on *SNLI test*, *NC test* and *RS test* sets.
- The predict command results in *testPredict_NC.txt* and *testPredict_RS.txt* files that contain the predictions on *NC test* and *RS test* sets.

## Sample Input
> {"sentence1": "Van beek is the director of non-proliferation and space in the Trade and Industry Department.", "premise": "Van beek is the director of non-proliferation and space in the Trade and Industry Department.", "premiseUF": [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], "hypothesisUF": [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], "gold_label": "contradiction", "sentence2": "Van beek is the director of non-proliferation and space in the Trade and Industry Department.", "hypothesis": "Van beek is the director of non-proliferation and space in the Trade and Industry Department."}

premiseUF and hypothesisUF are the list of Unary Feature vectors where each vector is a four dimensional vector. The four dimensions represent *Name of a Person/City/Country*, *Date Type Entity(Day of the week/Month/Year)*, *Numeric Entity (Cardinal/In Words)* and *Other* entity. 

## Additional Scripts
```
python utils/Add_NERFeatures_To_SNLIFormatted_Jsonl.py --file_path_name ./data/sample/testSample.jsonl
```
- This is script to convert a jsonl file with premise hypothesis sentences into a jsonl file with the Unary Feature vectors for premise and hypothesis. 
- The testSample.jsonl file is a sample file in the snli format. 
- Dependecies: Spacy and StanfordNER (download its zip file from https://drive.google.com/file/d/1q8hy3ZlxnURla0fFI8Ic9xc0Z05HxVbC/view?usp=sharing )
- Unzip the StanfordNER zip file.
- Update SAVE_DIR (where you would like to save the results) and STANFORDN_NER_DIR ( path to the unzipped folder)
- There are two outputs (available for reference in /data/sample/) from this script 1) a joblib file with only the NER feature vectors, 2) a jsonl file with the NER Feature vectors in the format of "Sample Input" shown above,
 
