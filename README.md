# Report: Baseline Study on Electrooculography Data
## Neurocognitive Computing
### David Dembinsky

## Overview
This reports evaluates three different models on the EOG dataset, a MLP, CNN and LSTM.
From the raw dataset, windows are formed, either by calculating statistical properties or concatentating all raw data that falls into a frame.

Note, that the KNN classifier is not included in the report, due to time and space constraints. However, it had good performance on the intra-person-prediciton, but was too slow and had too high memory requirments for the other setups. Tackling this issue or using better hardware it could be promissing.

## Structure
- ```data/```: Here the raw dataset is stored as provided by the EOG dataset.
- ```res/```: Saves output figures and log_files.
- ```src/```:
    - ```KNN.py```: A k-Nearest-Neigbours classifier, that is not included in the report.
    - ```model_evaluate.py```: Main File. Code to train and test a model on a dataset. Has functions to evaluate multiple models on each setup.
    - ```model_performance.py```: Used to track the number of parameters and train_time for each model.
    - ```models.py```: Contains the MLP, CNN and LSTM as well as a base class all inherit from.
    - ```preprocess_data.py```: Forms dataloaders from the EOG dataset by normalizing, performing PCA and calculatign windows.
- ```src/```:
    - ```dataset_statistics.ipynb```: Overview on the class distribution of the dataset and unused raw signal figure.
    - ```dataset.ipynb```: Plots of the datasets feature correlation and PCA plot.
    - ```model_performance.ipynb```: Plots the data obtained by ```model_performance.py``` 
    