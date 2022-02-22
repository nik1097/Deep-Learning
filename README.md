# Deep-Learning CPSC - 8430 
Project Repo for the course Deep Learning taught by Dr. Feng Luo.
There are 3 homework assignments that will be worked on throughout the semester. It covers a wide array of Deep Learning concepts. 

All the files in the repository are .ipynb files that contain python code snippets and results of executing the code in a sequential manner.

## Contents
- [Homework 1](https://github.com/nik1097/Deep-Learning/tree/master/Homework%201) contains python code that demonstrate some of the fundamental concepts in Deep Learning.
  - [Simulating Function](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20simulate%20function.ipynb)
    Simulated a sin function and created neural network models with multiple layers. 
    Concepts covered:
    - Creating and transforming Tensors
    - Loss Functions
    - Accuracy
    - Activation function
    - Epochs
  - [Training on MNIST dataset](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/Hw1%20comMNIST.ipynb)
    Built a neural network model to classify the MNIST dataset, plotted the graphs for loss and accuracy to understand the different types of concepts in the Neural Network:
    - Convolution Neural Networks
    - Fully connected layer - dense layer
    - Maxpool layer
    - Activation Function
    - Dropout, weight_decay
  - [Compare Parameters](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20param_compare.ipynb)
    Varied the number of parameters of a Neural Network model to understand the relation between number of parameters and accuracy/loss.
    - modifying parameters
  - [Random Labels](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20rand_label_fit.ipynb)
    Randomised training labels input to understand the effect of using random labels for training a Neural Network.
    - neural networks that memorize rather than learn
  - [Principal Component Analysis](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20PCA.ipynb)
    Applied PCA over the weights learn by the MNIST dataset to visualize the reduction of dimensions of the weights.
    - identified weights that are important for the model to learn
    - collection of weights of model and layers
  - [Sensitivity in Neural Networks](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20Flat%20vs%20General_sensitivity.ipynb)
    Trained 2 models on the MNIST dataset to understand the relation between sensitivity and batch size.
    - batch size
    - validated decrease in sensitivity after ideal batch size
  - [Interpolation](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20Flat%20vs%20General%20interpolation.ipynb)
    Interpolated the weights of 2 Deep Learning models to visualiaize the ability of the model to generalize.
    - generalization
    - interpolation
    - collection of weights
  - [Gradient Norm](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20grad_norm.ipynb)
    Observed the gradient norm during training and visualized the concept of Loss vs change in gradients(weights)
    - gradients

### Required Python Packages
- Pytorch
- pandas
- numpy
- sklearn (only for PCA)
- matplotlib
