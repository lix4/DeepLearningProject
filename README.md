# Tracking Block Operations

This project is part of Rose-Hulman's entry in the IBM AI XPRIZE Competition. The overall goal is to use AI to make it easier for non-technical people to use robots. The project is to use Deep Learning to automate the tracking of block movements from video. Over the summer we have constructed 9 wooden blocks that have expensive ($300) sensors embedded in them. The sensors can measure block acceleration and transmit these measurements to a laptop using bluetooth. We want to train a neural network to use video to predict the block accelerations so that video recording of the remaining blocks (which don't have the sensors) can be automatically tracked.

## Goal
The goal of this project is to use a video to determine if and how blocks in the video are moving. It is a prediction/regression project. The XPrize team needs to predict acceleration and orientation of those blocks without sensors. We are trying to use a combination of current videos, and sensor data from blocks to train a deep network.  After training, inputting a video into the network should give accelerometer and orientation information for each of the blocks visible on the screen without expensive sensors.

## Dataset
Our data was collected from trials that students did last summer. We have 6 folders of data with each one containing one video and one csv dataset for each of seven different blocks. After incorporating them into an integrated big dataset, we will get a very large dataset for training and validation.

![image](https://user-images.githubusercontent.com/12198981/32744545-d5da32a2-c87d-11e7-902e-6101416572ce.png)


## Dataset Analysis
The videos in each folder are used as input, with the sensor values as an output.  Each block’s related csv file includes eight features.  These features are: time stamp, time stamp unix, low noise acceleration X, low noise acceleration Y, low noise acceleration Z, gyroscope X, gyroscope Y, gyroscope Z.  Both the ‘time stamp’ and ‘time stamp unix’ are the same feature in different formats.  Additionally, the csv file should be synced with the input (video) in such a way that time is not needed as a feature at all.  This leaves the output to be a set of six features per block for a total of 42 features.  

## Data PreProcessing
Each video contains a start tone to indicate the beginning of data recording.  We trimmed each video to start at this tone so the sensor data and the video would align.  The videos were also downscaled to an image size of 240 by 240.  This drastically decreased the required memory and network size without losing too much information.  Since the video and sensor data were stopped manually, we trimmed the end of the data and videos to match lengths.

In total, we had 20520 frames worth of data (three channels each), and matching sensor output.  In order to preserve some of the time-related connections of the data, we created two sets of input frames.  The first (current) set is the original frames shifted back one.  The second (previous) set is the same as the original set with the end frame removed.  These two sets are then combined to create a 4D tensor of shape [20520, 2, 240, 240, 3].

We split up the data into testing and training.  The testing set contains the final 5000 data points.  This is longer than one entire test video of moving blocks.  Thus, the testing and training data are balanced as best as possible.  **BLANK**% of data was chosen for validation.
