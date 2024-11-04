# FTTP

## Introduction
This project is for the ICDE2025 paper, which leverages distributed hybrid intelligence to adapt arbitrary FL models in test time. Hybrid intelligence consists of human intelligence and model intelligence, which are important to update models.

## Datasets
- Human activity dataset:
  --'[Harth dataset](https://archive.ics.uci.edu/dataset/779/harth)'
  --'[PAMAP dataset](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)'
- Image classification dataset:
  --FMNIST dataset

## Features
- savemodels FOLDER: we pre-train FedAvg, FedProx and Ditto models and save the pre-trained model in the folder. Please load the pre-trained model before run test time adaptation code.
- Fedtest_top2.py is the main file to run the project, data is loaded by data_loader.py, and model architecture is loaded by model.py, update.py is used to update the pre-trained model using test data. The hyperparameters are set in utils.py
- data_process.py is used to pre-process the raw data.

## implementation
- Download the datasets.
- Pre-process the raw data with data_process.py. The process is used to divide the data into training sets for pre-trained models, samples for test time adaptation, and samples for validation in test time. Meanwhile, The process filters some useless features.
- Train the model from random initial models to obtain a pre-train model. And we store the pre-trained model in  the folder named savemodels
- Run Fedtest_top2.py, the parameters used for test time adaptation are stored in utils. The results can be accessed via Tensorboard.
