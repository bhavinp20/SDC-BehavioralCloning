# **Behavioral Cloning** 
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia_architecture.PNG  "Model Visualization"
[image2]: ./writeup_images/final_architecture.PNG   "Final model architecture"
[image3]: ./writeup_images/center_camera.jpg        "Center camera"
[image4]: ./writeup_images/left_camera.jpg          "Left camera"
[image5]: ./writeup_images/right_camera.jpg         "Right camera"

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

My model consists of a convolution neural network developed by NVIDIA. The model includes RELU layers to introduce 
nonlinearity, and the data is normalized in the model using a Keras lambda layer.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

#### 5. Solution Design Approach

1). I collected the data by driving the car in a simulator. I drove the cars in forward direction for 3 laps then I turn my car around to go in opposite direction for another 2 laps.

![alt text][image3]
![alt text][image4]
![alt text][image5]

2). I used OpenCV to load the images and convert it to RGB because the drive.py processes the images in RGB and since OpenCV reads the file in BGR format.

3). I split the collected data to 20% validation set and remaining 80% in training set.

4). I normalized the images.

5). I crop the image 70 pixels from top and 25 pixels from bottom. By cropping the image it removed the unwated pixels which was not necessary.

6). I then used the NVIDIA architecture for training my data.

7). I added a dropout to overcome overfitting of the model.

#### 6. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

![alt text][image2]
