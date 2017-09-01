# **Behavioral Cloning Project**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/model.png "Model Visualization"
[image2]: ./output_images/center_2017_09_01_08_34_51_333.jpg "Center"
[image3]: ./output_images/left_2017_09_01_08_34_51_333.jpg "Left"
[image4]: ./output_images/right_2017_09_01_08_34_51_333.jpg "Right"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `output_images/model.png` containing the model architecture generated from Keras visualize
* `README.md` or writeup_report.pdf summarizing the results
* `model.log` containing the log of the model training

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I experimented with a few simple models with 1 or 2 convolutional layers but decided to use the NVIDIA [model](https://arxiv.org/pdf/1604.07316v1.pdf) since it gave a good accuracy and eventually I was able to drive across the track with the model I trained using the architecture. My model consists of a convolution neural network with 5x5 / 3x3 filter sizes and depths between 24 and 64 for Convolutional layers (`model.py.build_model()`). The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. I used Keras Cropping2D layer to crop unnecessary parts of the image like the sky and the car hood.

The model contains dropout layers between the Dense layers in order to reduce overfitting. The model was trained and validated on different data sets by shuffling to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Their is still some jitter in the driving but all in all the vehicle drives without leaving the track and is able to recover if it ends up towards the edge after a turn.

The model used an Adam optimizer, so the learning rate was not tuned manually. Training data was chosen to keep the vehicle driving on the road.

### Model Architecture and Training Strategy

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because it was proven to perform in the paper mentioned earlier in the writeup. It had a decent number of convolutional layers as well as dense layers to train the model in a relatively short time and obtaining a good accuracy.

In order to gauge how well the model was working, I split my image and steering angle data into a training `70%` and validation set `30%`. I added a `ModelCheckpoint` with a callback option to save only model steps which reduced the overall validation loss. I found that my first model performed fairly well until it came to the sharp turn after the bridge and ended up running onto the dirt track on the right. To fix this I added the left and right camera data to the training set with a `0.2` steering correction. This fixed the issue on the aforementioned sharp turn. However, at the next right turn just before the end of the lap, the vehicle ran into water. I collected additional data for that section of the track and re-trained the model to fix it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle veered a little off of the center, not off the track, but was able to recover and drive back to the center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The final model architecture (`model.py.build_model()`) consisted of a convolution neural network with the following layers

| Layer                 |       Description                             |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 BGR image                           |
| Normalization Lambda  | Normalize the image                           |
| Cropping2D            | Crop (70, 25), outputs 65x320x3               |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 31x158x24  |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 14x77x36   |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, valid padding, outputs  5x37x48   |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, valid padding, outputs  3x35x64   |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, valid padding, outputs  1x35x64   |
| RELU                  |                                               |
| Flatten               | outputs 2112                                  |
| Dense                 | outputs 100                                   |
| RELU                  |                                               |
| Dropout               | Probability 0.25                              |
| Dense                 | outputs 50                                    |
| RELU                  |                                               |
| Dropout               | Probability 0.25                              |
| Dense                 | outputs 10                                    |
| RELU                  |                                               |
| Dropout               | Probability 0.25                              |
| Dense                 | outputs 1                                     |

Here is a visualization of the architecture generated using [Keras and PyPlot](https://faroit.github.io/keras-docs/1.2.2/visualization/)

![alt text][image1]

I used the following driving data to train the model:
* 2 Laps of driving in the center
* Quarter lap of recovery driving. Switched off the recording before veering off the track and started recording again when trying to recover to the center
* Some additional recording to around sharp corners since the vehicle was running into the dirt track

##### Sample images:
Center camera:

![alt text][image2]

Left camera:

![alt text][image3]

Right camera:

![alt text][image4]

The first model I trained used just center camera images. The model didn't perform well around corners. After adding left and right camera images and adding additional training data around corners it started getting better. After the collection process, I had `8115` data points.
```sh
carnd@ip-10-61-159-37:~$ wc -l data/driving_log.csv
8115 data/driving_log.csv
```
I tried preprocessing the data by applying a histogram equalizer to each of the channels but didn't get much improvement. So I removed the preprocessing step to train the model faster. I finally randomly shuffled the data set and put `30%` of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 since there wasn't much improvement after training for more than 5. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

Here's a link to my video result (Same as `./video.mp4` in the repository)

[![Behavioral-Cloning](http://img.youtube.com/vi/FYODl5XPvoE/0.jpg)](http://www.youtube.com/watch?v=FYODl5XPvoE "Behavioral-Cloning")
