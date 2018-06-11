# Robot-Football-Vision-AI
Developed for the PROJ 324 module as part of the third year of the MEng Robotics course at Plymouth University, this project uses a Convolutional Neural Network to identify the presence of size 3 footballs from a usb webcam feed. This repository contains code used in developement of the dataset, training the model from the dataset, and the machine vision end-application.


## About the programs
The following section details the programs 

### webcam_recorder.py
This program records videos from a webcam connected to the computer. Its use is for data acquisition; in this case filmnig footballs. It is heavily based on XXXLINK.

### big_data_prep.py
A very elegant solution to preparing a dataset from raw data. 

Given raw data in the form of videos, outputs a dataset of images ready for training. The videos are stored in folders of class (ball & no_ball). The filepaths for these folders is specified in the program. When executed, the program iterates through all the videos, and saving every Nth frame to folders of class.

So long as the raw data videos only feature the class. To explain, videos must always feature the ball in frame (ball), or not at all (no_ball)). These videos are placed in folders, and the with filepaths specified, the program will prepare an entire dataset.

For XXGB of raw video, a i5 @ 3GHz pc takes xx minutes to build a dataset of 110,000 images.


### aug_train.py
Used to train a model from a dataset of images.

The program makes use of the ImageDataGenerator keras function, to perfomr augmentation on the images when training.By applying various transformations to the images (rotation, shear???, width and height shift etc) the model is never presented with the exact same image twice. This reduces overfitting and helps the model generalize well the real world applications in new scenarios, as the model is forced to learn the underlying features and structures in the images that define the object. In example, humans can easily recognize a football in an image, even when the image has been distorted. By providing the model with augmented images it too can learn what makes a football a football, and avoid an "easy answer" such as the circular path of colour in the image.

While image augmentation is the crux of the models performance, other training parameters offer valuable performance improvements. A bathc size of xx, learning rate decay, epochs ????


Results in model.h5 containing XXX and model.json which contains XXX, and a weights.h5 file, which contains the weights used for the model. By saving the model X, network architecture, & weights sepeartly, different weights can be swapped for use in the machine vision applications.

### live_predict_location.py
The machine vision application used to demonstrate deployment of the model to mobile hardware, and test performance on unseen data (how the model performs in the real world).

The program uses video from a live webcam, and runs every Nth frame through the model. If the the predcition is of a ball, the program then runs the model on five different regions of the image (four corners, and a central region). The region with the best prediction is determined to be the location of the ball. Addtionally, the program features the option of using the openCV hough_circles function on the winning region to attempt a precise location of the ball.

### video_predict_location.py
As the implementation of threading makes uses videos as the input more difficult than it should be, this version of the program has the trheading stripped out, but otherwise does exactly the same as live_predict_location.py on saved videos.


# Results


## Videos
Videos of the application in action.


## invariance to location
reference videos and talk a bit

## invariance to obstruction

## invariance to enviroment

## invariance to lighting

## real-time performance


ADD IN DATA FOR ABOVE STUFF



# Future work
For anyone intending to use this work, here are the intentions for developing the projecct going forward.







