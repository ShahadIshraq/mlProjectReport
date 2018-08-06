[//]: # (Image References)

[image1]: ./pipeline.png "ASR Pipeline"
[image2]: ./nnArch.jpg "Architecture"



# Speech Recognition with Neural Networks

# Project for CSE472

Collector: Student ID
Coder: Student ID
Trainer: 1305023
Writer: 1305022
Leader: 1305002
# I. Definition



### 1.1. Project Overview

This project implements a deep learning nural network which can predict a transcription for an audio speech. A high level view of the pipeline is shown in the image below. 

![ASR Pipeline][image1]

The [LibriSpeech dataset](http://www.openslr.org/12/) has been used to train and evaluate the model.This contains a large corpus (About 1000 hours) of English-read speech from audiobooks. But only a small portion of the data has been used for this project. 


### 1.2. Problem Statement
The main task of this project was to run and modify  [this GitHub project](https://github.com/lucko515/speech-recognition-neural-network) where audio of a speech is taken as input and the texual inscription of the speech is given as output.
We have modularized the codes, tuned the hyperparameters and transformed the project to the deliverable format.

### 1.3. Performance Metrics
This project has used  CTC (Connectionist Temporal Classification) Loss as performance metric. CTC loss function is widely used for training recurrent neural networks (RNNs) such as LSTM networks that deal with sequence problems like online handwriting recognition and speech recognition, where the timing is variable.
We have not dug deep into the mathematical basis and algorithm behind this function and used the [ctc_loss function](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss) from Tensorflow.
And as the CTC loss function has been used in training the model, the ctc_decode function has been used on the output of softmax to generate the final output. 

# **II. Analysis**


### 2.1.  Data description
The dataset contained a vast array of labeled flac files that contained the speech of the text label. The flac files were first converted to wav. The wav files were then used for training and validating the model.
JSON-Line description files corresponding to the train and validation datasets were then created. 

### 2.3. Algorithms and Techniques
**Layers :**

- **Recurrent :** In this layer, connections between nodes form a directed graph along a sequence. This allows it to exhibit dynamic temporal behavior for a time sequence. 
- **Convolution :**  Use of this layer helps the network to learn filters that activate when it detects some specific type of feature at some spatial position in the input. 
- **Time Distributed Dense** : Dense layers are the traditional fully connected networks that maps the scores of the convolutional layers into the correct labels with some activation function(softmax used here). Time distributed dense layer is used to keep one-to-one relations on input and output. 

**Activation functions :** 

Activation layers apply a non-linear operation to the output of the other layers. 
- **ReLu Activation** : ReLu or Rectified Linear Unit computes the function $f(x)=max(0,x) to threshold the activation at 0.
- **Softmax Activation :** [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) is applied to the output layer to convert the scores into probabilities that sum to 1.

**Optimizers :**

**Batch Normalization :**  Batch Normalization tries to properly initializing neural networks by explicitly forcing the activations throughout a network to take on a unit gaussian distribution at the beginning of the training. We put the Batchnorm layers right after Dense or convolutional layers. Networks that use Batch Normalization are significantly more robust to bad initialization. Because normalization greatly reduces the ability of a small number of outlying inputs to over-influence the training, it also tends to reduce overfitting. 


# III. Model and Architecture

# 3.1. Model description :
In this specific problem, two possible feature representations could be used.
1. Spectograms
2. MFCC (Mel-Frequency Cepstral Coeffficients)

Spectrogram provides a visual representation of the spectrum of frequencies of sound as they vary with time. Spectrograms are extensively used in the fields of music, sonar, radar, and speech processing,etc.  
On the other hand,MFCCs are coefficients that collectively make up an MFC ,which is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. 
The  common method of derivation of MFCCs take the Fourier transform of a window of a signal, map the powers of the spectrum obtained by Fouruer transform onto the mel scale, take the logs of the powers at each of the mel frequencies, take the discrete cosine transform of the list of mel log powers and then taking the amplitudes of the resulting spectrum.
We have not dug deep into MFCC and have simply used a library function that provides the MFCC representation of a given audio.
The MFCC representation is of much lower dimensions than the Spectrograms. This provided us simplicity needed to emphasise on the wholistic view of the project instead.
The usable layers in case of speech recognition are :
1. **RNN :**  Time sequence audio features is input into this layer There can be 29 characters(letters with space,apostrophe and blank) as output for each time sequence of an audio file.
2. **RNN+TimeDistributed Dense :**  The addition of an extra dense layer can be helpful in recognizing more complex speech patterns.
3. **CNN+RNN+TimeDistributed Dense :**   An additional 1-D convolutional layer help in handling more complexity.
4. **Deeper RNN+TimeDistributed Dense :** Multiple layers of RNN can also be used.
5. **Bidirectional RNN+TimeDistributed Dense :** BRNNS processes data in both directions. This allows the use of context.
6. **CNN+DEEPER RNN+TimeDistributed Dense :** This is the final model we have used in our implementation of this project.


# 3.2. Architecture description :

The image below shows an overview of the final architecture we have used.

![Architecture][image2]

We have used softmax layer after the time distributed dense layer.
In training time , we take a batch of input and process the MFCC of the batch together. In the training process, ctc_loss function is used.When the model is used for prediction, the output of the softmax layer is fed into the ctc_decode function to produce text.
Also we have used batch normalization after convolution layer and time disributed dense layer.

# IV. Methodology

### 4.1. Data Preprocessing :
The audio files of the dataset are in flac format. A text file is provided to carry the transcriptions of the labels of each data file. 
At first, the flac files have been converted to wav format.
After that, JSON files corresponding to the train and validation datasets are created. The training and testing module use these JSON files as file-descriptor to access the data files and corresponding labels.


##4.2. Hyperparameter tuning :
We have performed tuning on three hyperparameters : 
1. Number of filters
2. Kernel size
3. Number of units
We varied the number of filters among 150,200 and 250, kernel_size among 7,11 and 15 and finally number of units among 150,200 and 250.
The *tune.py* module does the tuning part and outputs the result to hyperparameter.txt file.
    
# V. Result

# VI. Conclusion


