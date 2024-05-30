[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14936378&assignment_repo_type=AssignmentRepo)
# Cropped Word Recognition



[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)

[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14936378&assignment_repo_type=AssignmentRepo)

# Cropped Word Recognition

## Introduction
Our project focuses on the challenging task of recognizing text from cropped word images. The aim is to develop models that can accurately identify the text in images or perform "word spotting" by extracting lexical embeddings for retrieval purposes.

![alt text](https://i0.wp.com/theailearner.com/wp-content/uploads/2021/01/OCR_ICTC.png?w=495&ssl=1)


## Structure
Here is the structure of our github:


## Model Types
In this repository you'll find 2 different methods with which we have done cropped
word recognition.
- PHOCNet (Baseline)
- CRAFT (https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file)
  
## Baseline
This project has a baseline as a starting point for the development of the task.
The baseline is based in PHOCNet (Pyramidal Histogram Of Characters Network).
  
## Datasets
* **IIIT 5K**   
The IIIT 5K-word dataset is harvested from Google image search, we refer to it as the 'REAL' dataset. Query words like billboards, signboard, house numbers, house name plates, movie posters were used to collect images. The dataset contains 5000 cropped word images from Scene Texts and born-digital images. Although this dataset was recommended we ended up not using it because of time restrictions.

![alt text](https://cdn-xlab-data.openxlab.org.cn/cover/IIIT_5K/iiit5k.png) 
    
* **VGG**   
This dataset consists of **9 million images** covering **90k English words**. It is formed by words of varying fonts. It is a synthetic dataset and pictures are in grayscale, , we refer to ir as the 'Synthetic Dataset'  over the text. We ended up using **200k images** for training, **20k** for validation and **20k** for testing as recommended by the teachers in the tutoring sessions.
![alt text](https://i.imgur.com/1eA7tDt.png) 

* **Generated Dataset**   
We decided to generate our own dataset to deal with the PHOC baseline's poor performance. With white background, black letters and with different fonts. This is all in the repository under the name Generate Dataset. Here you can see some generated images examples, in the creation of the Dataset it is ensure that the test set is unseen data. We created **50k** images for training and **5k** for testing.
![alt text](https://i.imgur.com/mxGJh1s.png) 
  
Two primary datasets will be used for training and evaluation:

* *Synthetic Dataset*:  A dataset comprising 9 million of images.
* *Generated Dataset:*  A smaller, more challenging dataset with colourised words for evaluation on the PHOC baseline model.

### PHOC
#### PHOCNet Architecture
The model architecture we are using is shown in this first image. It features a series of convolutional layers, interspersed with max pooling layers after the first and second convolutions. Following these, there is a Spatial Pyramid Pooling (SPP) layer, which is included because it allows the CNN to handle input images of varying sizes while still producing a fixed-size output. This fixed-size output is essential for the fully connected layers, which require a consistent input size. The final layer of the architecture has 604 neurons, corresponding to the size of PHOC.
![alt text](https://imgur.com/unjt2Zn.png)
  

####  PHOC inner workings
PHOC works at the end like a retrieval. In the case of QbE, literally looking at the most similar-looking images and and retrieving that word. In the case of QbS, sending that image to a vector and retrieving the most similar vectors. 

When mapping a image to a vector PHOC does a binary representation of character strings encoding the presence of characters in different sections of a word. It utilises multiple levels of splits to capture character positions and frequent bigrams.

-   **Query-by-Example (QbE)**: Retrieves word images based on visual similarity to a given example image.
-   **Query-by-String (QbS)**: Retrieves word images based on a textual query representation, requiring a mapping from text to image features.

![alt text](https://i.imgur.com/uTKdoyf.png) 

### Implementation
Using the PyTorch code of PHOC (the original is in Caffe), an image can be processed to obtain its PHOC representation. The next step is to identify the corresponding word from this representation. This is accomplished using a KNN with the precomputed PHOC representations of 90,000 words. Thus, the image-to-text conversion process involves: first, passing the image through the PHOCNet to generate the representation; then, feeding this representation into the KNN, which will return the closest matching word  based on the image-derived representation.

Our whole implementation can be seen in our directory under the folder number PHOC, inside of it a Jupyter notebook with the pipeline and necessary auxiliary and wandb tracking functions.

### Model Performance
#### First Results
This model was trained using these set of parameters:
* *Loss*: BCEwithlogits
* *Schedulers*: StepLR
* *Optimizers*: SGD with Momentum

Training with the Synthetic Dataset and the previously designed parameters created really bad predictions as we can see here because of the loss getting stuck we receive mostly 1-character answers mostly 'e' because of how common it is in the lexicon.txt file. We detected problems in: the dataset we were using (as maybe it was too difficult for PHOC), our parameters and loss function. The accuracy was 0.

Here you can see some predictions of our firs approach:
![alt text](https://i.imgur.com/52gsIOF.jpeg)

#### Final Results
The loss was getting stuck and not really creating a good function to proceed to a better results so after looking at a previous implementation of PHOCNet on Github we adapted their implementation of a proper loss function to our code. The idea: to define the weights for an error as the inverse of the 'appeareance frequency' of the that word in the lexicon.txt file. This is done under a function called 'create_weights()' used in the make and training processes.

We also needed to start with a low learning rate (we set for 0.001) and use ReduceLROnPlateu to make sure we getting stuck in any local minima. The optimizer was changed to the Adam optimizer over SGD because it adapts the learning rate for each parameter, leading to faster convergence and improved performance on complex problems.

Updated parameters:
* *Loss*: BCEwithlogits
* *Schedulers*: ReduceLROnPlateau
* *Optimizers*: Adam

After overcoming quite a few problems with the training, the model ends up giving great results.  The loss did not get stuck and the accuracy for our test set was 0.9978, the edit distance 0.0028. We are happy with the results. In the next picture you can see some predictions done over the Generated Dataset.

![alt text](https://i.imgur.com/Xeqtt20.jpeg) 

Some predictions, we searched for errors to show them here:
![alt text](https://i.imgur.com/fEuFJTT.jpeg)



## Primer Approach

Vista la baseline, passarem al nostre primer enfoc per intentar resoldre el problema de reconeixement de paraules. Aquest enfoc consta de dos passos principals: detecció i reconeixement.

![Procés de detecció i reconeixement d'imatge](https://es.mathworks.com/help/vision/ocr-category.png)

El procés de detecció consisteix en localitzar i extreure els caràcters presents en la nostra imatge. Utilitzarem tècniques de detecció d'elements per identificar i delimitar cadascun dels caràcters presents a la imatge. Això ens permetrà tractar cada caràcter de manera individual i independent durant el procés de reconeixement. És important tenir en compte que amb aquest enfoc no estem tenint en compte el context de la paraula completa, ja que tractem cada caràcter com una entitat independent.

Un cop hem detectat els caràcters de la imatge, pasarem al procés de reconeixement, on mitjançant diferents models de CNN intentarem entrenar-los per reconèixer i assignar l'etiqueta correcta a cada caràcter individual.

### Detection

Comencarem abordant el problema de la detecció dels caràcters de la paraula amb la que estiguem treballant. Explorarem dos metodologies principals per dur a terme la detecció: la primera consisteix en utilitzar un sistema de reconeixement òptic de caràcters (OCR) mitjançant tècniques de visió per computador, i la segona es basa en el model YOLO (You Only Look Once), més concretament el YOLOv8, que és actualment un dels millors en el camp de la detecció. 


## Example Code

The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site) package to monitor how your network is learning, or not.

  

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

- Marino Oliveros Blanco NIU: 1668563
- Luis Domene García NIU: 1673659
- Joan Bayona Corbalán NIU: 1667446

Deep Learning & Neural Networks
Artificial Intelligence degree
UAB, 2024
