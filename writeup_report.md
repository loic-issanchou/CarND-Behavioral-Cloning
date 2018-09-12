# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed 

The model includes RELU layers to introduce nonlinearity (code line 117->122), and the data is normalized in the model using a Keras lambda layer (code line 115). 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 125, 128, 131, 134). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 138->140). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, clockwise and counter-clockwise. I not used images recovering from the left or the right but it might be a very good idea for improve accuracy of the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to determines steering angle from images.

My first step was to use a convolution neural network model. I thought convolution neural network model might be appropriate because it allow to classifie images depends on the steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it include dropout layer.

Then I obtain a mean squared error on the validation set which is not high but which not decrease epoch after epoch. So my model don't allow to generalize ths behavioral of the car well.

The final step was to run the simulator to see how well the car was driving around track one. There was no spots where the vehicle fell off the track but like I said just before, the model can be largely improve to generalize the behavioral on all the possible tracks.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road but only the track one ! The vehicle is not able to drive autonomously around the track two.

#### 2. Final Model Architecture

The final model architecture (model.py lines 115->135) consisted of a convolution neural network with the following layers and layer sizes which are detailed below.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


| Layer         		|     Description	        									| 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 70x200x3 YUV and normalized image 							| 
| Convolution 5x5     	| 24 output filters, "relu" activation and (2,2) subsamples    	|
| Convolution 5x5		| 36 output filters, "relu" activation and (2,2) subsamples    	|
| Convolution 5x5	   	| 48 output filters, "relu" activation and (2,2) subsamples    	|
| Convolution 3x3	    | 64 output filters, "relu" activation							|
| Convolution 3x3		| 64 output filters, "relu" activation							|
| Flatten		      	|  																|
| Dropout				| keep_prob=0.5 training set   									|
| Dense					| outputs 100  													|
| ELU					|																|
| Dropout				| keep_prob=0.5 training set    								|
| Dense					| outputs 50  													|
| ELU					|																|
| Dropout				| keep_prob=0.5 training set    								|
| Dense					| outputs 10  													|
| ELU					|																|
| Dropout				| keep_prob=0.5 training set    								|
| Dense					| outputs 1  													|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I use data provided by Udacity which use center lane driving. 

To augment the data sat, I also flipped images and angles thinking that this would help not overfit the data. The track turns globally on the left, so when flipping images, it allows to include a track that turns globally on the right. 


I then preprocessed this data by convert images in YUV color space, cropping and resize images. Finally, I normalized the data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. Less than 5 is a risk of under fitting and more is a risk of over fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.