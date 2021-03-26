**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Sample_Train_Images.jpg "Sample Train Images"
[image2]: ./Model_Train_History.jpg "Model Training History"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral_cloning.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model3.h5 containing a trained CNN model
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model3.h5
```

#### 3. Submission code is usable and readable

The behavioral_cloning.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with filter sizes 7, 5, and 3 and depths between 16 and 256 (behavioral_cloning.ipynb cell # 6) 

The model includes RELU layers to introduce nonlinearity, Dropout layers for regularization, and the data is mean normalized in the model using a Keras lambda layer. The input image is cropped using the Keras Cropping2D layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (behavioral_cloning.ipynb cell # 6). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (behavioral_cloning.ipynb cell # 7). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (behavioral_cloning.ipynb cell # 7). Number of Epochs hyperparameter was tuned to achive best validation metric performance.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used mostly center lane driving data. I collected training data by running two laps on the first track - one CW and another CCW - Both laps were clean driving data & no off-lane recovery data was collected. I combined this data with the data provided in the project Repo. I didn't explicitly take the recovery from off-lane driving data, as the model was able to do well on the first track just with the clean driving data. For the first track, the model is essentially seeing the train data itself in the autonomous mode. That's why the model was able to do well even without the off-lane recovery data. However, data corresponding to recovery would be needed for the model to be able to generalize well on unseen tracks. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a CNN model and train until the train set performance saturates; i.e. the bias is removed, then regularize the model to improve the validation set performance by removing variance. Subsequently, test the model on the simulator and if the model is not at all able to drive the car then increase the complexity of the model, else if the model is mostly able to drive the car and fails in certain places then collect more data for those places.

My first step was to use a 6 layer deep CNN model with 
* filter kernel sizes 7, 5, and 3 
* three Dense layers, and 
* no Dropout layers 
Then i trained this model and tuned the number of Epochs to maximize the validation set performance (MAE on both train & validation sets around 0.05). When I tested this model on the simulator, the car was mostly able to drive but it was oscillating between the left & right lane lines in a sinusoidal manner but at the same time remaining within the lane lines. At some occassions, the car went off road as well.

Then I increased the model complexity to total 9 layers and added two Dropout layers to handle overfitting and tuned the model for best validation test performance. This time the MAE on both train & validation set were around 0.08, which was higher that what I got on the previous smaller model. However, this model did better on the test track and the oscillation behavior was gone. The model was able to nicely drive the car on the first track. I tested this model on second track but it went off the road after driving correctly for a while. This shows that the model is not generalizable to unseen tracks, but it is doing well on the first track.


#### 2. Final Model Architecture

The final model architecture (behavioral_cloning.ipynb cell # 6) consisted of a convolution neural network with the following layers and layer sizes - 

|       Layer (type)                  |      Output Shape         |      Param #        |
|:-----------------------------------:|:-------------------------:|:-------------------:| 
|   lambda (Lambda)                   |   (None, 160, 320, 3)     |      0              |
|   cropping2d (Cropping2D)           |   (None, 90, 320, 3)      |      0              |
|   conv2d (Conv2D)                   |   (None, 84, 314, 16)     |      2368           |
|   max_pooling2d (MaxPooling2D)      |   (None, 42, 157, 16)     |      0              |
|   conv2d_1 (Conv2D)                 |   (None, 36, 151, 32)     |      25120          |
|   max_pooling2d_1 (MaxPooling2D)    |   (None, 18, 75, 32)      |      0              |
|   conv2d_2 (Conv2D)                 |   (None, 14, 71, 64)      |      51264          |
|   max_pooling2d_2 (MaxPooling2D)    |   (None, 7, 35, 64)       |      0              |
|   conv2d_3 (Conv2D)                 |   (None, 5, 33, 128)      |      73856          |
|   max_pooling2d_3 (MaxPooling2D)    |   (None, 2, 16, 128)      |      0              |
|   conv2d_4 (Conv2D)                 |   (None, 2, 16, 256)      |      33024          |
|   flatten (Flatten)                 |   (None, 8192)            |      0              |
|   dense (Dense)                     |   (None, 512)             |      4194816        |
|   dropout (Dropout)                 |   (None, 512)             |      0              |
|   dense_1 (Dense)                   |   (None, 256)             |      131328         |
|   dense_2 (Dense)                   |   (None, 128)             |      32896          |
|   dense_3 (Dense)                   |   (None, 64)              |      8256           |
|   dense_4 (Dense)                   |   (None, 1)               |      65             |
|:-----------------------------------:|:-------------------------:|:-------------------:| 

* Total params: 4,552,993
* Trainable params: 4,552,993
* Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving - once CW and once CCW. Here is an example image of center lane driving:

![alt text][image1]

I combined this data with the data provided in the Project repo.
After the collection process, I had 10,134 data points. I, then, added / subtracted a correction of 0.2 for the steering angles for images from left and right cameras respectively. This led to total 30,402 data points. I split this dataset 80:20 into train and validation sets.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 13 as evidenced by the following training history plots.

![alt text][image2]

