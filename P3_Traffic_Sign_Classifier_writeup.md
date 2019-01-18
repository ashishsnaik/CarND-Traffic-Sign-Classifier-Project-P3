# **Traffic Sign Recognition** 

## Traffic Sign Recognition Program using Convolutional Neural Network (ConvNet) in TensorFlow
---

**The goals / steps of this project are the following:**

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./dataset_class_distribution.png "Class Distribution"
[image2]: ./lenet_ep25_kp50_accuracy.png "LeNet Accuracy"
[image3]: ./signnet1_ep25_kp50_accuracy.png "SignNet1 Accuracy"
[image4]: ./signnet2_ep25_kp50_accuracy.png "SignNet2 Accuracy"
[image5]: ./signnet3_ep25_kp50_accuracy.png "SignNet3 Accuracy"
[image6]: ./signnet3_refined_ep25_kp50_accuracy.png "Final SignNet3 Accuracy"
[image7]: ./downloaded_images.png "Downloaded Images"
[image8]: ./downloaded_traffic_sign_images/04.jpg "Speed30 Image"
[image9]: ./sample_train_images.png "Sample Training Images"
[image10]: ./sample_val_images.png "Sample Validation Images"
[image11]: ./sample_test_images.png "Sample Test Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

Here are links to my [project code](Traffic_Sign_Classifier.ipynb) and the code's [HTML output](Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and matplotlib libraries to calculate summary statistics and check the class distributions of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are a few exploratory visualizations of the data set. 

**10 Sample images (with Labels) from the Training Set:**

![alt text][image9]

**5 Sample images (with Labels) from the Validation Set:**

![alt text][image10]

**5 Sample images (with Labels) from the Test Set:**

![alt text][image11]


**A bar chart showing how the Classes in the Training, Validation, and Test sets are distributed:**

![alt text][image1]

There is quite some class imbalance here, as expected! But, the imbalance trend seems to be somewhat similar across the *Validation* and *Test* sets. Assuming that the Test set is a representative of the real world data, validating the model on this Validation set, which has a similar distribution, should give a sensible model.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The road signs, along with the actual 'sign' itself, do have color as one of the differentiating factors for different types of signs. Hence, I did not change the images to grayscale, and used the 3-channel RGB image in order to preserve the color diferrentiation.

The only preprocessing I used is normalizing the image data, so the data are centered and scaled.

I decided to try the model training without augamenting the data and checking whether I would need to add more data in order to take care of overfitting. The Dropouts technique gave me fairly good result in reducing overfitting, hence I did not used augmented data. Though, in the next version of the project, which I plan after this submission, I would certainly use data augmentation in order to expose my model to additional data during training.  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried 4 different model architectures, the first being the LeNet with 3 channel input and the other 3 being variations of the LeNet with different number of convolution filters and their width. I also used SAME padding for one of the architectures (`SignNet3`) and VALID padding for the others. The 4 models, namely `LeNet`, `SignNet1`, `SignNet2`, and `SignNet3`, I experimented with can be found in my code `Traffic_Sign_Classifier.ipynb` notebook (or `Traffic_Sign_Classifier.html`) under section *Model Architecture* sub-sections *LeNet* and *SignNets*.

Based on the performances of the 4 models (more on this in the later sections below), I decided to use the `SignNet3` model as the final one.

My final model `SignNet3` consisted of the following 6 layer architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 				    |
| Flatten       	    | outputs 2048 									|
| Dropout       	    |           									|
| Fully connected		| outputs 512 									|
| RELU					|												|
| Dropout       	    |           									|
| Fully connected		| outputs 256 									|
| RELU					|												|
| Dropout       	    |           									|
| Fully connected		| outputs 43 									|
| Softmax				| outputs 43        							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The initial training of my 4 models can be found in my code `Traffic_Sign_Classifier.ipynb` notebook (or `Traffic_Sign_Classifier.html`) under section *Train, Validate and Test the Model* sub-sections *SignNet3*, *SignNet2*, *SignNet1*, and *LeNet*.

For the initial training and selection of the model, I used Adam Optimizer, 25 epochs, batch size of 64, and a learning rate of 0.001 for all the 4 models, and keep_prob of 0.50 for the SignNet models.

Later, after selecting the `SignNet3` model for final training, I used Adam Optimizer, 30 epochs, keep_prob of 0.50, batch size of 64, and switched the learning rate from 0.001 to 0.0005 after 20 epochs. 

The best Training and Validation accuracies (in %) during the initial training were as below.

**SignNet3:** Best Training Accuracy = 99.92 ; Best Validation Accuracy = **98.03**

**SignNet2:** Best Training Accuracy = 99.86 ; Best Validation Accuracy = 96.89

**SignNet1:** Best Training Accuracy = **99.97** ; Best Validation Accuracy = 97.78

**LeNet:** Best Training Accuracy = 99.95 ; Best Validation Accuracy = 94.20


**Training and Validation Accuracy Charts**

LeNet                      |SignNet1
:-------------------------:|:-------------------------:
![][image2]                |![][image3]


SignNet2   |SignNet3
:-------------------------:|:-------------------------
![][image4]                |![][image5]


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As we can see in the above section, with sufficient epochs, all the models reached a validation accuracy above 93%, but my goal was to get as close to 100% as possible. Though, as seen from the accuracies and charts, all models seem to be overfitting a bit (training accuracy > validation set). Of course, getting more data, with data augmentation, or using say L2 regularization would help here, though, the overfitting (or variance) problem here doesn't seem to be a very critical issue with the given scores. So, I selected the `SigNet3` model, which shows the least overfitting and has the highest validation accuracy.

The final training code can be found in `Traffic_Sign_Classifier.ipynb` notebook (or `Traffic_Sign_Classifier.html`) under section `Refine the best model and Test`. As mentioned above, for the final training, I used Adam Optimizer, 30 epochs, keep_prob of 0.50, batch size of 64, and switched the learning rate from 0.001 to 0.0005 after 20 epochs, which gave be the below results.

My final model results were:
* training set accuracy of 99.99%
* validation set accuracy of 97.53% 
* test set accuracy of 96.03%

**Final SignNet3 Training and Validation Accuracy Chart**
![alt text][image6]

I started with the LeNet architecture and thought it would be a better idea to preserve the 3 color channels. Keeping in mind the VGG16 network architecture that I have used for other image recognition projects, to work with the possible added complexity with Road Signs, as compared to handwritten digits, I changed the convolutions to 3x3 (from 5x5), to mark the finer details in the images. I also added a little more depth to the network with additional conv layer, and added more number of filters in the conv layers. I also made the fully connected layers denser, all this so I could first overfit and get good accuracy on the traning set and later added dropouts to ensure that the network also generalizes well. In the next version of the implementation, I certainly plan to experiment with data augmentation and L2 regularization, along with some possible modifications to the network architecture itself and train the network for more epochs.

I also tried with a batch size of 128 and other values of keep_prob ranging from 0.4 to 0.7, but found batch size of 64 and keep_prob of 0.5 to be giving better results, with the number of epochs I was running.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I ran this portion of the project on 10 German Traffic Sign images that I downloaded from the web and which can be found in the [downloaded traffic sign images]('./downloaded_traffic_sign_images/') folder.

Here are the 10 German traffic signs that I found on the web:
![alt text][image7]

The downloaded images have different resolutions and I used the OpenCV library to resize them to 32x32 resolution. The  interpolation used for resizing the images might distort the images and they might be difficult to classify. That could be the reason that resized image 04.jpg above, which is a speed limit 30 km/h sign and is very clear in the original image (below), was mis-classified as speed limit 50 km/h. Also the backgrounds and their distortions due to resizing in some of the images could add to the difficulty to classify. 

**Original non-resized Image 04.jpg**

![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image	   | Result  |Actual	        		               | Prediction	        		             | 
|:--------:|:-------:|:---------------------------------------:|:---------------------------------------:| 
| 01.jpg   | Correct | 25-Road work   					       | 25-Road work   						 | 
| 02.jpg   | Correct | 11-Right-of-way at the next intersection| 11-Right-of-way at the next intersection| 
| 03.jpg   | Correct | 12-Priority road                        | 12-Priority road                        | 
|**04.jpg**|**Wrong**| **1-Speed limit (30km/h)**              | **2-Speed limit (50km/h)**              | 
| 05.jpg   | Correct | 12-Priority road                        | 12-Priority road                        |
| 06.jpg   | Correct | 38-Keep right                           | 38-Keep right                           | 
| 07.jpg   | Correct | 14-Stop                                 | 14-Stop                                 | 
| 08.jpg   | Correct | 33-Turn right ahead                     | 33-Turn right ahead                     |
| 09.jpg   | Correct | 13-Yield                                | 13-Yield                                |
| 10.jpg   | Correct | 36-Go straight or right                 | 36-Go straight or right                 |


The model was able to correctly predict 9 of the 10 traffic signs, which gives an accuracy of 90%. The model's Test set accuracy was 96.03%, so this prediction is favourable as compared to the accuracy on the test set.

The implementation can be found in `Traffic_Sign_Classifier.ipynb` notebook (or `Traffic_Sign_Classifier.html`) under `Step 3: Test a Model on New Images` under sub-section `Predict the Sign Type for Each Image`.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located under `Step 3: Test a Model on New Images`.

For the first image, 01.jpg, the model is 100% sure that this is a road work sign (probability of 1.0), and the image does contain a road work sign. The top five softmax probabilities for that image were

| Probability (%)     	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.0         		| 25-Road work   								| 
| 0.0     				| 28-Children crossing 							|
| 0.0					| 11-Right-of-way at the next intersection		|
| 0.0	      			| 20-Dangerous curve to the right				|
| 0.0				    | 18-General caution      						|


For the fourth image (04.jpg), which is a 1-Speed limit (30km/h) and for which the prediction was wrong, the top five softmax probabilities were 

| Probability (%)     	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 45.74         		| 2-Speed limit (50km/h)   						| 
| 40.34     			| 1-Speed limit (30km/h) 						|
| 4.56					| 14-Stop		                                |
| 4.33	      			| 4-Speed limit (70km/h)				        |
| 2.29				    | 0-Speed limit (20km/h)      					|


As an obervation, the true class of the sign was identified as the 2nd most probable sign with 40.34% probability, which is close to the top (wrong) score. Also, all the top 5 probabilites scores show traffic signs that are close in structure and color combination.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


