# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. All required files 

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py containing helper functions to generate data for training and validation.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of  strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.(model.py lines 50-54) . The five convolutional layers followed with three fully connected layers, leading to a final output steering.(model.py lines 58-60)

The model includes RELU layers to introduce nonlinearity (code line 50-60), and the data is normalized in the model using a Keras lambda layer (code line 47). 

#### 2. Attempts to reduce overfitting in the model

I tried to use the dropout layers but I found that it is not very helpful for my model. So I decided not to use dropout. Instead, I use early stoping to reduce overfitting. (model.py lines 70). 

![image-20180612210040441](https://ws2.sinaimg.cn/large/006tKfTcly1fs8p1be5t7j30fi0arwfh.jpg)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 25-34). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving forward on track one and track two, backward on track two.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet. I think I need to do some pre-processing on the raw image. I normalized the data and mean centered the data. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

After several epochs training. The car can't even reach the first curve. It always tends to turn left. I realize that the data I feed to the model have more left turn. So I flipped images and steering measurements to help with the left turn bias. 

The simulator captures images from three cameras mounted on the car: a center, right and left camera. The image below gives a sense for how multiple cameras are used to train a self-driving car. This image shows a bird's-eye perspective of the car. The driver is moving forward but wants to turn towards a destination on the left. From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera. From the right camera's perspective, the steering angle would be larger than the angle from the center camera. 

![Angle between the destination and each camera](https://s3.cn-north-1.amazonaws.com.cn/u-img/998f4ea8-ae42-41d1-a6aa-b5ff3af995e1)

​				Angle between the destination and each camera(image from udacity)

By using the multiple cameras, the car can get to the first turn.It's better but not enough.

I found that not all of these pixels contain useful information, however. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. So I crop the lower and higher parts of the image to ignore the hood (bottom 20 pixels), sky/hills/trees (top 70 pixels)

Then I tried to use a more powerful model designed by nvidia.  The car sometimes drives towards the side of the road. Though it can recovery by itself by made a sharp turn, it's behavior is too dangerous. So I collected more data focusing on driving smoothly around curves in the opposite direction. I also used data from track two.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at speed 20 mph.

#### 2. Final Model Architecture

The final model architecture (model.py lines 35-61) is shown below.

------

| Layer (type)              | Output Shape        | Param # |
| ------------------------- | ------------------- | ------- |
| lambda_1 (Lambda)         | (None, 160, 320, 3) | 0       |
| cropping2d_1 (Cropping2D) | (None, 70, 320, 3)  | 0       |
| conv2d_1 (Conv2D)         | (None, 33, 158, 24) | 1824    |
| conv2d_2 (Conv2D)         | (None, 15, 77, 36)  | 21636   |
| conv2d_3 (Conv2D)         | (None, 6, 37, 48)   | 43248   |
| conv2d_4 (Conv2D)         | (None, 4, 35, 64)   | 27712   |
| conv2d_5 (Conv2D)         | (None, 2, 33, 64)   | 36928   |
| flatten_1 (Flatten)       | (None, 4224)        | 0       |
| dense_1 (Dense)           | (None, 100)         | 422500  |
| dense_2 (Dense)           | (None, 50)          | 5050    |
| dense_3 (Dense)           | (None, 10)          | 510     |
| dense_4 (Dense)           | (None, 1)           | 11      |

Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![center_2016_12_01_13_31_12_937](data/forward/IMG/center_2016_12_01_13_31_12_937.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive itself to the center of the track. These images show what a recovery from  left side of the track looks like.

![WX20180613-141711](../../Downloads/WX20180613-141711.png)

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help with the left turn bias. For example, here is an image that has then been flipped:

![WX20180613-142220](../../Downloads/WX20180613-142220.png)

After the collection process, I had 24654 number of data points. 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. I then preprocessed this data by normalizing the data and mean centering the data and cropping the the image. The validation set helped determine if the model was over or under fitting. The number of epochs was decided by early stopping. I used an adam optimizer so that manually training the learning rate wasn't necessary.
