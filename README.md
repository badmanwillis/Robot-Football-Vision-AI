# Robot-Football-Vision-AI
Developed for the PROJ 324 module as part of the third year of the MEng Robotics course at Plymouth University, this project uses a Convolutional Neural Network to identify the presence of size 3 footballs from a usb webcam feed. This repository contains code used in developement of the dataset, training the model from the dataset, and the machine vision end-application.


## About the programs
The following section details the programs 

### webcam_recorder.py
This program records videos from a webcam connected to the computer. Its use is for data acquisition; in this case filmnig footballs. It is heavily based on the OpenCV tutorial ["Getting started with videos"](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html).

### big_data_prep.py
A very elegant solution to preparing a dataset from raw data. 

Given raw data in the form of videos, outputs a dataset of images ready for training. The videos are stored in folders of class (ball & no_ball). The filepaths for these folders is specified in the program. When executed, the program iterates through all the videos, and saving every Nth frame to folders of class.

So long as the raw data videos only feature the class. To explain, videos must always feature the ball in frame (ball), or not at all (no_ball)). These videos are placed in folders, and the with filepaths specified, the program will prepare an entire dataset.

For XXGB of raw video, a i5 @ 3GHz pc takes xx minutes to build a dataset of 110,000 images.


### aug_train.py
Used to train a model from a dataset of images.

The program makes use of the ImageDataGenerator keras function, to perfomr augmentation on the images when training.By applying various transformations to the images (rotation, shear???, width and height shift etc) the model is never presented with the exact same image twice. This reduces overfitting and helps the model generalize well the real world applications in new scenarios, as the model is forced to learn the underlying features and structures in the images that define the object. In example, humans can easily recognize a football in an image, even when the image has been distorted. By providing the model with augmented images it too can learn what makes a football a football, and avoid an "easy answer" such as the circular path of colour in the image.

While image augmentation is the crux of the models performance, other training parameters offer valuable performance improvements. A bathc size of xx, learning rate decay, epochs ???? were all implemented.

After training, the models architecture is saved to a .json file, and the model weights are saved to a .h5 file. By saving the models network architecture & weights separately, different weights can be swapped for use in the machine vision applications.

### live_predict_location.py
The machine vision application used to demonstrate deployment of the model to mobile hardware, and test performance on unseen data (how the model performs in the real world).

The program uses video from a live webcam, and runs every Nth frame through the model. If the the predcition is of a ball, the program then runs the model on five different regions of the image (four corners, and a central region). The region with the best prediction is determined to be the location of the ball. Addtionally, the program features the option of using the openCV hough_circles function on the winning region to attempt a precise location of the ball.

### video_predict_location.py
As the implementation of threading makes uses videos as the input more difficult than it should be, this version of the program has the threading stripped out, but otherwise does exactly the same as live_predict_location.py on saved videos.


## Videos
Videos of the project through its development.


## CNN MV application success
Object recognition achieved in early March, with an ovefit model (dataset of less than 700 samples), to test code. [Link - CNN MV application success](https://www.youtube.com/watch?v=br7uWylh_Wc)

## Image Dataset labelling tool.
Intended as a tool to speed up dataset labelling of position coordinates, this program had intended use in localisation from the model approach. [Link - Image Dataset Labelling Tool](https://www.youtube.com/watch?v=btqyVI-VSBE). This was abandoned when the decision was made to focus on localisation via approximation, instead of via the network.

## invariance to obstruction

## invariance to enviroment

## invariance to lighting
[Proj324 Results: Lighting in-variance](https://www.youtube.com/watch?v=O8hVivgu7Ws&t=7s). Shows how the model handles significant changes in lighting conditions, and continues to perform admirably even in extreme conditions. As the model is intended for use in a controlled lighting enviroment, the lighting conditions in the video were deliberately intended to push the model to its limits.


## real-time performance
The model averages close to 30fps when a ball is not detected, but slows down to 2fps when localising the ball. This is due to a poor software implementation of localisation. Most of the project time was spent researching localisation from a network, and once that was scrapped, limited time was available to develop a high performance region approximation application. Improvements are discussed as part of future work, problems with the current approach are listed below.

* Program is written in python, not C.
* Program saves images from the webcam to a folder, then reads those images back in to make predictions. This is a tedious approach, but making predictions


# Future work
For anyone intending to use this work, here are the intentions for developing the projecct going forward.







