# Project Skyhawk

The project aims to create a system that performs classification and clustering tasks on provided data in order to detect automatically generated texts or texts containing misinformation. The dectetion is based on Natural Language Processing (NLP) techniques and Machine Leanring. 

## Installation

Dowload the code and unzip it in a folder that will be project's folder. On Linux system run **setup.sh** script as root user to setup virtual environment and install all necessary dependencies from **requirements.txt**. Attention: the script upgrades *virtualenv*, *pip*, and *setuptools* packages.


## Code structure

Source code in **src** folder contains 2 modules:
1) *Classification* (**classification_*.py**) to predict if a given article is real or disinformation
2) *Clustering* (**clustering.py**) to divide disinformation articles into fine grained clusters

Classification module contains two separate scripts: 
1) **classification_train.py** - trains chosen classifiers on training data, selects the best on validation data (portion of training data), and saves its model
2) **classification_inference.py** - loads best model and applies it to new data to classify

File **config.py** contains code to load a configuration file and setup a module. Files **data_loading.py**, **feature_extraction.py**, and **model.py** contain code for 1) loading and pre-processing input data, 2) extracting features, and 3) training models, getting predictions and evaluating performance, respectively. 


## Configuration

Every module has its own configuration file containing paths to input and output data, as well as other parameters. These configuration files have corresponding names with **.ini** extensions and are situated in **configs** folder.

Before running the modules, paths to input data and output models should be set up in the configuration files instead of YOUR_PATH tokens. Quotes are not necessary. Values of other fields can also be changed to change the model and its results.


## Additional data

If word embeddings are used (`[features] word_embeddings = True` in configuration file), then a path to pre-trained *Word2Vec* embeddings should be provided in word_embeddings.path. Original model uses *Word2Vec* embeddings trained on *GoogleNews* corpus with ~100B words and contain 300-dimensional vectors. They can be downloaded from [https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). The file has to be extracted before use. 


## Run a module

Follow the following instruction to run a module:
1) change directory to project folder 
2) edit configuration file
3) activate virtual environment<br />
 `source env/bin/activate`
4) run module script<br />
ex: `python src/clustering.py`
 5) deactivate virtual environment<br />
 `deactivate`


## Deploy classification system

To deploy a classification system, first run **classification_train.py** module and save results (done automatically but don't forget to provide output path in the configuration file). Then, to get predictions on new articles run **classification_inference.py** module.

