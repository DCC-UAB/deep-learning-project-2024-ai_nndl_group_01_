[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14936378&assignment_repo_type=AssignmentRepo)
# Cropped Word Recognition
## Introduction

Our project focuses on the challenging task of recognizing text from cropped word images. The aim is to develop models that can accurately identify the text in images or perform "word spotting" by extracting lexical embeddings for retrieval purposes.

## Objectives
The primary goal is to enhance text recognition capabilities using advanced machine learning techniques. This involves:

* Recognizing and accurately transcribing cropped word images.
* Extracting lexical embeddings to facilitate efficient word retrieval.

## Model Types

To achieve these goals, we will experiment with various types of models:

* Convolutional Neural Networks (CNN): Utilized for their effectiveness in image processing tasks.
* CNN combined with Recurrent Neural Networks (CNN + RNN): Leveraging RNNs for sequential data handling alongside CNNs.
* Transformers: Applying cutting-edge models known for their prowess in natural language processing tasks.

## Task

The core task is text recognition, where the models will be trained to discern and transcribe text from images of cropped words.
Data

Two primary datasets will be used for training and evaluation:

* Synthetic Data: A large dataset comprising 9 million images of 90,000 words.
* Real Data: A smaller, more challenging dataset with 5,000 words.

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors
Write here the name and UAB mail of the group members

Xarxes Neuronals i Aprenentatge Profund
Grau de __Write here the name of your estudies (Artificial Intelligence, Data Engineering or Computational Mathematics & Data analyitics)__, 
UAB, 2023
