# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

[Simulator](https://github.com/udacity/self-driving-car-sim) for collecting data and autonomous driving.

### Details About Files

My project includes the following files:

* model.py (create a model to predict steering angle, output a file`model.h5`used by `drive.py`)
* utils.py (helper functions for load and generate the data)
* model.h5 (contain a trained model for predicting steering angle)
* drive.py (use`python drive.py model.h5` to drive the car in autonomous mode)
* writeup_report.md (summarizing the reports)
* run1.mp4 (drive autonomously on track one)
* run2.mp4 (drive autonomously on track two)



for more details about the project see `writeup_report.md`