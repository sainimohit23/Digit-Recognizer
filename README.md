# Digit Recognizer

This program lets you recognize digits using a neural network trained on MNIST dataset.


### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

### Description
Program is currently set to recognize digits written with blue color. `TF_MNIST_v4.0.py` file trains NN for 20 epochs and saves the trained weights in checkpoints directory.

`recognizer.py` uses trained weights to classify digits. It uses openCV to find bounding boxes and it uses a 2 layered neural network for classification of digits.

* Classifies digits in range 0-9
* Train accuracy of neural network : ~99
* Dataset used : MNIST

Structure of Neural Network used in program:

![nn](https://user-images.githubusercontent.com/26195811/46398138-eaa73c00-c711-11e8-9539-0321bf3343d1.png)


### Execution for writing through webcam
Run `recognizer.py` and input number of digits you want to recognize.



## Sample images:


![threshold_img](https://user-images.githubusercontent.com/26195811/46398720-7a99b580-c713-11e8-856c-4bb5733880bb.jpg) ![output_img](https://user-images.githubusercontent.com/26195811/46398743-8d13ef00-c713-11e8-9bcd-ffb474f2fe8a.jpg)




![gif](https://user-images.githubusercontent.com/26195811/46398911-00b5fc00-c714-11e8-970d-c459a67c5a1a.gif)
