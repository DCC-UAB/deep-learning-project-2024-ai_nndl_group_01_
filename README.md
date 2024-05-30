[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14936378&assignment_repo_type=AssignmentRepo)
# Cropped Word Recognition

## Introduction

Our project focuses on the challenging task of recognizing text from images. The aim is to develop models that can accurately identify the text in images.

Our project focuses on two different tasks detection and recognition. The detection part consist of identifying in an image where the text is. Recognition is the part that takes the part of the image where the word is and identify what text is in the image.

![alt text](https://i0.wp.com/theailearner.com/wp-content/uploads/2021/01/OCR_ICTC.png?w=495&ssl=1)


## Structure
Here is the structure of our github:
```
├───File Splitting
│	├───Samples
│	├───file-splitting-GENERATED_DATASET.ipynb
│	├───file-splitting-RealYesVal.ipynb
│	└───file-splitting-Synthetic.ipynb
├───Generate Dataset
│	├───Fonts
│	├───Generated_Samples
│	└───Dataset_Generator.ipynb
├───PHOC
│	├───PHOC_model.ipynb
│	├───bigrams.txt
│	├───first
│	└───lexicon.txt
├───detection/CRAFT/CRAFT-pytorch
│	├───basenet
│	├───results
│	├───LICENSE
│	├───README.md
│	├───analysis_results_own.ipynb
│	├───craft.py
│	├───craft_utils.py
│	├───file_utils.py
│	├───imgproc.py
│	├───refinenet.py
│	├───requirements.txt
│	├───single_image_test_modified.py
│	└───test.py
├───recognition
│   ├───CRNN_model.ipynb
│   ├───best_model.pth
│   ├───test_data.csv
│   ├───test_res.ipynb
│   ├───train_data.csv
│   ├───val_data.csv
│   └───visualizations.ipynb
├───src
├───.gitignore
├───LICENSE
├───README.md
├───environment.yml
```

## Model Types
In this repository you'll find 3 different methods that help us in our task:

* Detection:
	- CRAFT (https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file)
* Recognition:
	- PHOCNet (Baseline)
	- CRNN
  
## Baseline
This project has a baseline as a starting point for the development of the task.
The baseline is based in PHOCNet (Pyramidal Histogram Of Characters Network).
  
  
## Datasets
* *IIIT 5K*   
The IIIT 5K-word dataset is harvested from Google image search, we refer to it as the 'REAL' dataset. Query words like billboards, signboard, house numbers, house name plates, movie posters were used to collect images. The dataset contains 5000 cropped word images from Scene Texts and born-digital images. Although this dataset was recommended we ended up not using it because of time restrictions.

![alt text](https://cdn-xlab-data.openxlab.org.cn/cover/IIIT_5K/iiit5k.png) 
    
* *VGG*   
This dataset consists of *9 million images* covering *90k English words. It is formed by words of varying fonts. It is a synthetic dataset and pictures are in grayscale, , we refer to ir as the 'Synthetic Dataset'  over the text. We ended up using **200k images* for training, *20k* for validation and *20k* for testing as recommended by the teachers in the tutoring sessions.
![alt text](https://i.imgur.com/1eA7tDt.png) 

* *Generated Dataset*   
We decided to generate our own dataset to deal with the PHOC baseline's poor performance. With white background, black letters and with different fonts. This is all in the repository under the name Generate Dataset. Here you can see some generated images examples, in the creation of the Dataset it is ensure that the test set is unseen data. We created *50k* images for training and *5k* for testing.
![alt text](https://i.imgur.com/mxGJh1s.png) 
  
*  *ICDAR 2013*   
The ICDAR 2013 dataset is a benchmark dataset used for evaluating algorithms for robust reading, containing a variety of images with text in natural scenes, annotated with ground truth for text localization and recognition tasks.
![alt text](https://i.imgur.com/ZhF5amL.jpeg)

Two primary datasets will be used for training and evaluation:


* Synthetic Dataset:  A dataset comprising 9 million of images.
* Generated Dataset:  A smaller, more challenging dataset with colourised words for evaluation on the PHOC baseline model.
* ICDAR 2013 used for detection algorithms evaluation.

### PHOC
#### PHOCNet Architecture
The model architecture we are using is shown in this first image. It features a series of convolutional layers, interspersed with max pooling layers after the first and second convolutions. Following these, there is a Spatial Pyramid Pooling (SPP) layer, which is included because it allows the CNN to handle input images of varying sizes while still producing a fixed-size output. This fixed-size output is essential for the fully connected layers, which require a consistent input size. The final layer of the architecture has 604 neurons, corresponding to the size of PHOC.
![alt text](https://imgur.com/unjt2Zn.png)
  

####  PHOC inner workings
PHOC works at the end like a retrieval. In the case of QbE, literally looking at the most similar-looking images and and retrieving that word. In the case of QbS, sending that image to a vector and retrieving the most similar vectors. 

When mapping a image to a vector PHOC does a binary representation of character strings encoding the presence of characters in different sections of a word. It utilises multiple levels of splits to capture character positions and frequent bigrams.

-   *Query-by-Example (QbE)*: Retrieves word images based on visual similarity to a given example image.
-   *Query-by-String (QbS)*: Retrieves word images based on a textual query representation, requiring a mapping from text to image features.

![alt text](https://i.imgur.com/uTKdoyf.png) 

### Implementation
Using the PyTorch code of PHOC (the original is in Caffe), an image can be processed to obtain its PHOC representation. The next step is to identify the corresponding word from this representation. This is accomplished using a KNN with the precomputed PHOC representations of 90,000 words. Thus, the image-to-text conversion process involves: first, passing the image through the PHOCNet to generate the representation; then, feeding this representation into the KNN, which will return the closest matching word  based on the image-derived representation.

Our whole implementation can be seen in our directory under the folder number PHOC, inside of it a Jupyter notebook with the pipeline and necessary auxiliary and wandb tracking functions.

### Model Performance
#### First Results
This model was trained using these set of parameters:
* Loss: BCEwithlogits
* Schedulers: StepLR
* Optimizers: SGD with Momentum

Training with the Synthetic Dataset and the previously designed parameters created really bad predictions as we can see here because of the loss getting stuck we receive mostly 1-character answers mostly 'e' because of how common it is in the lexicon.txt file. We detected problems in: the dataset we were using (as maybe it was too difficult for PHOC), our parameters and loss function. The accuracy was 0.

Here you can see some predictions of our firs approach:
![alt text](https://i.imgur.com/52gsIOF.jpeg)

#### Final Results
The loss was getting stuck and not really creating a good function to proceed to a better results so after looking at a previous implementation of PHOCNet on Github we adapted their implementation of a proper loss function to our code. The idea: to define the weights for an error as the inverse of the 'appeareance frequency' of the that word in the lexicon.txt file. This is done under a function called 'create_weights()' used in the make and training processes.

We also needed to start with a low learning rate (we set for 0.001) and use ReduceLROnPlateu to make sure we getting stuck in any local minima. The optimizer was changed to the Adam optimizer over SGD because it adapts the learning rate for each parameter, leading to faster convergence and improved performance on complex problems.

Updated parameters:
* Loss: BCEwithlogits
* Schedulers: ReduceLROnPlateau
* Optimizers: Adam

After overcoming quite a few problems with the training, the model ends up giving great results.  The loss did not get stuck and the accuracy for our test set was 0.9978, the edit distance 0.0028. We are happy with the results. In the next picture you can see some predictions done over the Generated Dataset.

![alt text](https://i.imgur.com/Xeqtt20.jpeg) 

Some predictions, we searched for errors to show them here:
![alt text](https://i.imgur.com/fEuFJTT.jpeg)

##  Recognition
Considering that we have to perform two tasks we decided to use a pretrained model in recognition and implement our model from scratch in the recognition part. We considered to use two different models:
-  **EAST (Efficient and Accurate Scene Text Detector)**: EAST uses a fully convolutional network to predict word boxes and word orientations with high efficiency..
-   **CRAFT (Character Region Awareness for Text detection)**: CRAFT focuses on detecting individual character regions and links them to form words, providing precise localization of text.

![Procés de detecció i reconeixement d'imatge](https://es.mathworks.com/help/vision/ocr-category.png)

CRAFT (Character Region Awareness for Text detection) stands out among the word detection techniques due to several reasons:

1.  **Character-Level Detection**: Unlike many methods that detect words or text lines directly, CRAFT detects individual characters, allowing for more precise localization and better handling of various fonts and styles.
2.  **Robustness**: It performs well in detecting text in diverse environments, including complex backgrounds and different languages, making it versatile for real-world applications.
3.  **High Accuracy**: The character-level detection and robust linking mechanism contribute to higher accuracy in word detection compared to methods that operate at the word or line level.

In conclusion, CRAFT was chosen for its superior ability to detect characters accurately and link them effectively to form words, providing robust and precise word detection in various challenging scenarios.
![alt text](https://i.imgur.com/wuWEB4M.jpeg)


In order to check how the pre-trained model works we did the following. First we installed everything we need following the indications of https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file. Then we adapt some minors pieces of code that were obsolete and create scripts (single_image_test_modified.py) to apply the model on specific images and folders. Finally we use the code to generate the results of images of the test set of Challenge2_Task1_ in the ICDAR 2013 dataset.  Here we can observe the results:

![alt text](https://i.imgur.com/RtJWpj3.jpeg)
### Analysis of Results

The plot and the reported metric (Overall Mean IoU: 0.7540) represent the performance of an object detection system, specifically in terms of the overlap between the predicted bounding boxes and the ground truth annotations.

What is IoU? Intersection over Union (IoU) is a metric used to evaluate the accuracy of an object detection algorithm. It is 
#### Variability in IoU:

The plot shows significant variability in IoU values for different images. Some images have IoU values close to 1, indicating very accurate detections. Some images have low IoU values (even close to 0), indicating issues with detection for those specific images. Overall Mean IoU (0.7540):

The average IoU value is approximately 0.754, indicating that, in general, the detections have a good level of overlap with the ground truth annotations. A value of 0.754 is reasonably good in many object detection contexts but may not be sufficient in applications requiring extremely high precision.

#### Possible Causes of Variability in IoU

Variability in Images: Different levels of lighting, image angles, and complex backgrounds can affect the algorithm's accuracy. Size and Shape of Objects: Very small objects or objects with irregular shapes can be more challenging to detect accurately.

The images that performed the least effectively with the CRAFT model have some common characteristics that likely contributed to the lower performance:

1.  **Non-Standard Fonts and Characters**:
    
2.  **Reflective Surfaces and Glare**:
    
3.  **Background Complexity**:
    
4.  **Size and Orientation of Text**:
    ![alt text](https://i.imgur.com/xaSVfnU.jpeg)


### Recognition

Once we have finished the first part, now we can focus in text recognition itself. 



#### Text Recognition Pipeline 

Once text detection identifies text regions, these regions are processed through a text recognition pipeline:

1.  **Convolutional Layers:** Extract features from the cropped text regions.
2.  **LSTM Network:** Processes the features to generate sequential data.
3.  **CTC Decoder:** Converts LSTM outputs into readable text.

![merge](https://i.imgur.com/VBmrH9q.jpeg)

#### Receptive Fields 

The receptive field of a convolutional neural network (CNN) is the region in the input image that affects a particular feature in the output. As we move deeper into the CNN layers, the receptive field increases, allowing the network to capture more complex patterns.

#### CNN Features to LSTM Model 

Feature maps extracted by the CNN are reshaped and fed into an LSTM model. A feature map of shape (Batch_Size, 512, 1, 31) is reshaped to (Batch_Size, 31, 512), where:

-   31 corresponds to the number of time steps.
-   512 is the number of features per time step.

This sequential data is then processed by the LSTM to produce softmax probabilities over the vocabulary.
![merge](https://i.imgur.com/Bd29USv.jpeg)

#### Calculating Loss 

In text recognition tasks, ground truth is not available for every time step, making the use of categorical cross-entropy loss infeasible. Instead, Connectionist Temporal Classification (CTC) loss is used.

#### CTC (Connectionist Temporal Classification) to the Rescue 

CTC loss enables training without the need for alignment between input images and their corresponding text. The CTC decode operation involves merging repeated characters and removing blank characters to produce the final text.

**CTC Decode Steps:**

1.  Merge repeated characters.
2.  Remove blank characters.


![merge](https://i.imgur.com/x3ERRMt.jpeg)



-   **Components**:
    -   **CNN Layers**: Extract spatial features from input images.
        -   Multiple `Conv2d` layers with `SELU` activation and `MaxPool2d` for down-sampling.
        -   `BatchNorm2d` layers to normalize activations and stabilize training.
    -   **RNN Layer**: LSTM layers to capture temporal dependencies.
        -   Bidirectional `LSTM` with 2 layers, providing forward and backward context.
    -   **Fully Connected Layer**: Maps the RNN output to the number of classes.

**Initialization**:

-   **Purpose**: Ensures stable and effective training.
-   **Methods**:
    -   `kaiming_normal_` for weights initialization.
    -   `constant_` for bias initialization.

#### Hyperparameters

-   **Learning Rate**: `0.0003`
    -   **Purpose**: Controls the step size during optimization. A smaller value ensures gradual convergence.
-   **Weight Decay**: `0.0005`
    -   **Purpose**: Adds L2 regularization to prevent overfitting.
-   **Batch Size**: `512`
    -   **Purpose**: Number of samples per gradient update, affecting training stability and memory usage.
-   **Number of Epochs**: `25`
    -   **Purpose**: Total iterations over the training dataset to ensure model convergence.

#### Optimizers and Learning Rate Schedulers

-   **Optimizer**: `Adam`
    -   **Purpose**: Efficient and adaptive optimizer that adjusts learning rates for each parameter, improving convergence speed.
-   **Learning Rate Scheduler**: `ReduceLROnPlateau`
    -   **Purpose**: Reduces learning rate when a metric (validation loss) has stopped improving, helping in fine-tuning the training process.

#### Training and Evaluation

-   **Training**:
    -   **Loss Function**: `CTCLoss`
        -   **Purpose**: Suitable for sequence prediction tasks where the alignment between input and output sequences is unknown.
    -   **Gradient Clipping**: Prevents exploding gradients by capping the gradients during backpropagation.
-   **Evaluation**:
    -   **Decoding**: `ctc_decode` function translates the model’s output probabilities into text sequences.
    -   **Accuracy Calculation**: Computes both word-level and character-level accuracy to evaluate model performance.

#### Tools

-   **WandB**:
    -   **Purpose**: Experiment tracking and visualization tool to log metrics, losses, and model parameters.
    -   **Usage**: Logs training progress, validation metrics, and model checkpoints.

![merge](https://i.imgur.com/iUw7QsI.jpeg)

### Results

### Interpretation of Metrics

* Word Accuracy

This metric represents the proportion of predicted words that exactly match the ground truth words. A higher value indicates that more words are predicted correctly.

* Character Accuracy
This metric measures the percentage of characters in the predicted words that match the characters in the ground truth words. It shows how accurate the model is at the character level, even if the entire word is not correct.

 
* Average Levenshtein Distance
This metric indicates the average number of edits (insertions, deletions, or substitutions) needed to transform a predicted word into the corresponding ground truth word. A lower value means the predicted words are more similar to the ground truth words.

**Test Word Accuracy: 0.54** 
**Test Character Accuracy: 0.76** 
**Test Average Levenshtein Distance: 0.97**


## Contributors
Write here the name and UAB mail of the group members

- Marino Oliveros Blanco NIU: 1668563
- Luis Domene García NIU: 1673659
- Joan Bayona Corbalán NIU: 1667446

Deep Learning & Neural Networks
Artificial Intelligence degree
UAB, 2024
