# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output_images/Train_bar.jpg "Visualization"
[image4]: ./Online_picture/1.jpg "Traffic Sign 1"
[image5]: ./Online_picture/2.jpg "Traffic Sign 2"
[image6]: ./Online_picture/3.jpg "Traffic Sign 3"
[image7]: ./Online_picture/4.jpg "Traffic Sign 4"
[image8]: ./Online_picture/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/notebooks/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over the 43 types pf traffic signs.

Below is the bar graph pf the training data set. 
Bar graphs for validation and test data set are under the folder "Output_images".

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because I want to standardize the data set. I chose the normalization function provided in the problem description:(pixel - 128)/ 128, because it's a simple and easy way to normalize an image. After normalization, the pixel value range changed from 0~255 to -1~1.

As a second step, I decided to convert the images to grayscale because color of the image does not impact the decognition of traffic signs. And grayscaled image only has one color channel, which is smaller and easier to proces than RGB images (3 channels).


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten   			| Output: 400									|
| Fully connected		| Output: 200									|
| RELU					|												|
| Fully connected		| Output: 84									|
| RELU					|												|
| Fully connected		| Output: 43									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an learning rate of 0.001, number of epochs of 100, and batch size of 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.945
* test set accuracy of 0.933

If a well known architecture was chosen:
* What architecture was chosen? 
  I chose the LeNet5 archietecture.
* Why did you believe it would be relevant to the traffic sign application?
  Because LeNet5 is useful for classify classes of images. Waht we learnt from class is to use leNet5 to classify digits into 10 different catagories, which is essential same type of problem as classify the traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  I was able to get >93% accuracy in test set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because after resize it down to 32x32, the feature of the "animal" become less distingushed.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic signals 		| Traffic signals								| 
| Slippery Road			| Slippery Road									|
| Wild animals crossing	| Wild animals crossing							|
| No passing      		| No passing 					 				|
| Speed limit (60km/h)	| Speed limit (50km/h) 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.933.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

For the first image, the model is almost 100% sure that this is a traffic signals (probability of 1.0), and the image does contain a traffic signal. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Traffic Signals								| 
| 0.0     				| General caution								|
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Speed limit (30km/h)			 				|
| 0.0				    | Speed limit (50km/h)							|


For the second image, the model is almost 100% sure that this is a Slippery Road (probability of 1.0), and the image does contain a Slippery Road. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery Road 								| 
| 0.0     				| Dangerous curve to the left					|
| 0.0					| Right-of-way at the next intersection			|
| 0.0	      			| Dangerous curve to the right      			|
| 0.0				    | Beware of ice/snow            				|

For the third image, the model is almost 100% sure that this is a Wild animals crossing (probability of 1.0), and the image does contain a Wild animals crossing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery Road 								| 
| 0.0     				| Road work                 					|
| 0.0					| Slippery road                     			|
| 0.0	      			| Speed limit (30km/h)                			|
| 0.0				    | Speed limit (20km/h)            				|

For the fourth image, the model is almost 100% sure that this is a No passing (probability of 1.0), and the image does contain a No passing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No passing    								| 
| 0.0     				| End of no passing            					|
| 0.0					| No entry                          			|
| 0.0	      			| Dangerous curve to the right         			|
| 0.0				    | Slippery road                 				|

For the fifth image, the model is almost 100% sure that this is a Speed limit (50km/h) (probability of 1.0), but the image is actually a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (50km/h)							| 
| 0.0     				| Speed limit (60km/h)      					|
| 0.0					| Speed limit (30km/h)               			|
| 0.0	      			| Speed limit (80km/h)               			|
| 0.0				    | End of speed limit (80km/h)      				|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


