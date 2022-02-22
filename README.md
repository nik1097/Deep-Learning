# Deep-Learning CPSC - 8430 
Project Repo for the course Deep Learning taught by Dr. Feng Luo.
There are 3 homework assignments that will be worked on throughout the semester. It covers a wide array of Deep Learning concepts. 

All the files in the repository are .ipynb files that contain python code snippets and results of executing the code in a sequential manner.

## Contents
- [Homework 1](https://github.com/nik1097/Deep-Learning/tree/master/Homework%201) contains python code that demonstrate some of the fundamental concepts in Deep Learning.
  - [Simulating Function](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20simulate%20function.ipynb)
    Simulated a sin function and created neural network models with multiple layers.
    - Creating and transforming Tensors
    - Loss Functions
    - Accuracy
    - Activation function
    - Epochs
    <p float="center">
      <img src="https://user-images.githubusercontent.com/20815651/155138234-d2d63591-78ed-41e7-8efa-668f9ebb77c0.png" width="400" />
      <img src="https://user-images.githubusercontent.com/20815651/155138305-e59b65ee-ebc0-47da-bc18-9c4854de43c5.png" width="400" /> 
    </p>
  
  - [Training on MNIST dataset](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/Hw1%20comMNIST.ipynb)
    Built a neural network model to classify the MNIST dataset, plotted the graphs for loss and accuracy to understand the different types of concepts in the Neural Network:
    - Convolution Neural Networks
    - Fully connected layer - dense layer
    - Maxpool layer
    - Activation Function
    - Dropout, weight_decay
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155139787-6c8449ee-c5ac-43eb-884e-7d90c91c0533.png" width="400" />
      <img src= "https://user-images.githubusercontent.com/20815651/155139886-57a31a98-b5ee-4299-b0bc-1dd9c0234ff0.png" width="400" /> 
    </p>
  - [Compare Parameters](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20param_compare.ipynb)
    Varied the number of parameters of a Neural Network model to understand the relation between number of parameters and accuracy/loss.
    - modifying parameters
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155140112-fa1739e2-8e81-4b2e-863c-c9bd110c8b1c.png" width="400" />
      <img src= "https://user-images.githubusercontent.com/20815651/155140148-52027976-f524-4bd7-8156-422b99078356.png" width="400" /> 
    </p>
  - [Random Labels](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20rand_label_fit.ipynb)
    Randomised training labels input to understand the effect of using random labels for training a Neural Network.
    - neural networks that memorize rather than learn
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155140687-2412db44-6504-4056-8d4c-f02e92a05119.png">
    </p> 
  - [Principal Component Analysis](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20PCA.ipynb)
    Applied PCA over the weights learn by the MNIST dataset to visualize the reduction of dimensions of the weights.
    - identified weights that are important for the model to learn
    - collection of weights of model and layers
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155140772-22967359-3bd8-4836-b715-2277862d6475.png" width="400" />
      <img src= "https://user-images.githubusercontent.com/20815651/155140857-5eefb49d-b4e5-4f34-9b20-fc50faff7f22.png" width="400" /> 
    </p>
  - [Sensitivity in Neural Networks](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20Flat%20vs%20General_sensitivity.ipynb)
    Trained 2 models on the MNIST dataset to understand the relation between sensitivity and batch size.
    - batch size
    - validated decrease in sensitivity after ideal batch size
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155140995-d8f61554-93a4-438f-b3ac-e0357b59fb30.png" width="400" />
      <img src= "https://user-images.githubusercontent.com/20815651/155141034-874b3219-5552-43b6-ba5d-484056448432.png" width="400" /> 
    </p>
  - [Interpolation](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20Flat%20vs%20General%20interpolation.ipynb)
    Interpolated the weights of 2 Deep Learning models to visualiaize the ability of the model to generalize.
    - generalization
    - interpolation
    - collection of weights
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155141149-f021b267-5b78-4fdd-aa2e-e2ec520cee2f.png" width="400" />
      <img src= "https://user-images.githubusercontent.com/20815651/155141202-1b9cfbc3-3b09-42c1-be66-262ea1141581.png" width="400" /> 
    </p>
  - [Gradient Norm](https://github.com/nik1097/Deep-Learning/blob/master/Homework%201/HW1%20grad_norm.ipynb)
    Observed the gradient norm during training and visualized the concept of Loss vs change in gradients(weights)
    - gradients
    <p float="center">
      <img src= "https://user-images.githubusercontent.com/20815651/155141502-1c4e48e0-e993-4c43-b61b-5a629cf5a36c.png" />
    </p>

### Required Python Packages
- Pytorch
- pandas
- numpy
- sklearn (only for PCA)
- matplotlib
