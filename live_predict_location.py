'''
Olly Smith 2018
olly.smith1994@gmail.com

This application makes live predictions for the location of a ball, using region proposal on a trained CNN model, from a webcam. Threading was used to improve the performance of the webcam stream.
'''

import time
import threading
from threading import Thread
import cv2
import os
import sys
import numpy as np
import glob


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
# model.save option
from keras.models import load_model
from keras.preprocessing import image
# Data preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import model_from_json

# specify where frames should be saved for prediction purposes
#path = '/home/olly/projects/machine_learning/april/big_data_train/data/video/image'
path = '/home/olly/Desktop/may_demo/big_data/data/video/image'
loc_path = '/home/olly/Desktop/may_demo/big_data/data/loc_images/image' # for locations




class WebcamVideoStream:
	def __init__(self, src=1):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
 
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
 
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
 
	def read(self):
		# return the frame most recently read
		return self.frame
 
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True



# LOAD MODEL
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights-best-0.96.h5")

loaded_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Loaded model from disk")

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        '/home/olly/Desktop/may_demo/big_data/data/video/',
        target_size=(150, 150),
        batch_size=16,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

loc_generator = test_datagen.flow_from_directory(
        '/home/olly/Desktop/may_demo/big_data/data/loc_images/',
        target_size=(150, 150),
        batch_size=16,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels


def predict():
	#print 'prediction'
	predict_start = time.time()
	# Make prediction from said folder
	prob = loaded_model.predict_generator(test_generator, 1)
	

	#print "floats"
	#print float(prob[0])
	conf = 1 - prob[0]
	print('Object recognition Probability %.8f' % conf)

	if float(prob[0]) < 0.5:
		print "ball!"
		#cv2.putText(img, ('ball'),(10,100), font, 2.5,(255,255,255),3)
		cv2.putText(img, ('Ball'),(20,85), font, 2,(0,255,0),3)
		cv2.rectangle(img, (0,0), (150,150), (0,255,0), 5)
		
		res = cv2.resize(img, (150, 150))
		cv2.imshow('Object Recognition',res)
	
		'''			Localisation here			'''


		# Crop parts of the image & resize
		# start Y & end Y coords, then same for X
		crop_TL = frame[0:240, 0:320]
		#cv2.imshow("TL", crop_TL)
		crop_TR = frame[0:240, 320:640] 
		#cv2.imshow("TR", crop_TR)
		crop_BL = frame[240:480, 0:320] 
		#cv2.imshow("BL", crop_BL)
		crop_BR = frame[240:480, 320:640] 
		#cv2.imshow("BR", crop_BR)
		crop_C = frame[120:360, 160:480] # center region
		#cv2.imshow("C", crop_C)

		# Run them through the network

		# save images
		cv2.imwrite(os.path.join(loc_path , '01_TL.jpg'), crop_TL)
		cv2.imwrite(os.path.join(loc_path , '02_TR.jpg'), crop_TR)
		cv2.imwrite(os.path.join(loc_path , '03_BL.jpg'), crop_BL)
		cv2.imwrite(os.path.join(loc_path , '04_BR.jpg'), crop_BR)
		cv2.imwrite(os.path.join(loc_path , '05_C.jpg'), crop_C)

		loc_prob = loaded_model.predict_generator(loc_generator, 1)
		print "Region Probabilities"		
		print loc_prob

		# Compare probs
		#Toms nonsense hotfix for region detection
		loc_prob_list = [loc_prob[4],loc_prob[0],loc_prob[2],loc_prob[3],loc_prob[1]]
		#loc_prob_list = [loc_prob[0],loc_prob[1],loc_prob[2],loc_prob[3],loc_prob[4]]
		#print loc_prob_list
		#max_val, max_idx = max((val, idx) for (idx, val) in enumerate(loc_prob_list))
		#print "MAX val: %d, idx: %d" % (max_val, max_idx)
		min_val, min_idx = min((val, idx) for (idx, val) in enumerate(loc_prob_list))
		print "MIN val: %d, idx: %d" % (min_val, min_idx)
		Rconf = 1 - min_val
		print('Object Region Probability %.8f' % Rconf)

		# Pick winner

		loc_img_list = [crop_TL,crop_TR,crop_BL,crop_BR,crop_C]
		#cv2.imshow("max winner", loc_img_list[max_idx])
		#cv2.imshow("min winner", loc_img_list[min_idx])
		region = cv2.resize(loc_img_list[min_idx], (150, 150))
		


		if min_idx == 0:
			cv2.rectangle(preview, (0,0),(320,240), (0,255,0), 5)
			cv2.putText(region, ('TOP LEFT'),(20,75), font, 0.65,(0,255,0),1)
			print "Ball is in the TOP LEFT region"
		elif min_idx ==1:
			cv2.rectangle(preview, (320,0), (640,240), (0,255,0), 5)
			cv2.putText(region, ('TOP RIGHT'),(20,75), font, 0.65,(0,255,0),1)
			print "Ball is in the TOP RIGHT region"
		elif min_idx ==2:
			cv2.rectangle(preview, (0,240), (320,480), (0,255,0), 5)
			cv2.putText(region, ('BOTTOM LEFT'),(2,75), font, 0.65,(0,255,0),1)
			print "Ball is in the BOTTOM LEFT region"
		elif min_idx ==3:
			cv2.rectangle(preview, (320,240), (640,480), (0,255,0), 5)
			cv2.putText(region, ('BOTTOM RIGHT'),(2,75), font, 0.65,(0,255,0),1)
			print "Ball is in the BOTTOM RIGHT region"
		elif min_idx ==4:
			cv2.rectangle(preview, (160,120), (480,360), (0,255,0), 5)
			cv2.putText(region, ('CENTER'),(30,75), font, 0.65,(0,255,0),1)
			print "Ball is in the CENTER region"

		#cv2.rectangle(preview, (0,0), (640,480), (0,255,0), 20)
		predict_region = time.time()
		region_time = predict_region - predict_start
		print "Time to estimate region = %.2f" % region_time
		cv2.imshow("Object Localisation", region)		
		cv2.imshow('Olly Smith Robot Football AI',preview)
		#cv2.waitKey(100)

		'''			Hough Circles here			'''
		'''
		
		# HoughCircles to find ball
		hough = loc_img_list[min_idx]
		#cv2.imshow('hough',hough)
		gray = cv2.cvtColor(hough, cv2.COLOR_BGR2GRAY)
		gray = cv2.medianBlur(gray,5)
		# these are vals to tweak
		dp = 2# basically sensitivity

# dp - Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
		minDist = 5


		circles = 	cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist)
 		#		cv2.HoughCircles(image, method, dp, minDist)


		#print "\ncircles"
		#print circles
		#cv2.imshow('hough', gray)
		#cv2.waitKey(0)



		xTotal =0
		yTotal =0
		rTotal =0
		xAv = 0
		yAv = 0
		rAv = 0
		avCount = 0

		# ensure at least some circles were found
		if circles is not None:
			# convert the (x, y) coordinates and radius of the circles to integers
			circles = np.round(circles[0, :]).astype("int")
		 
			# loop over the (x, y) coordinates and radius of the circles
			for (x, y, r) in circles:
				# draw the circle in the output image, then draw a rectangle
				# corresponding to the center of the circle
				cv2.circle(gray, (x, y), r, (0, 255, 0), 4)
				cv2.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		 

				#average x,y,r
				avCount += 1
				xTotal += x
				yTotal += y
				rTotal += r
			
			xAv = xTotal / avCount
			yAv = yTotal / avCount
			rAv = rTotal / avCount
			
			#print ("xAv: %d, yAv: %d, rAv: %d" % (xAv, yAv, rAv))

			# show the output image
			#cv2.imshow('hough', gray)
			
			avCount = 0

			# Estimate X,Y from area


			if min_idx == 0:
				cv2.circle(preview, (xAv, yAv), rAv, (0, 255, 0), 4)
				cv2.rectangle(preview, (xAv - 5, yAv - 5), (xAv + 5, yAv + 5), (0, 128, 255), -1)

			elif min_idx ==1:
				#cv2.rectangle(preview, (320,0), (640,240), (0,255,0), 5)
				cv2.circle(preview, (xAv+320, yAv), rAv, (0, 255, 0), 4)
				cv2.rectangle(preview, (xAv+320 - 5, yAv - 5), (xAv+320 + 5, yAv + 5), (0, 128, 255), -1)


			elif min_idx ==2:
				#cv2.rectangle(preview, (0,240), (320,480), (0,255,0), 5)

				cv2.circle(preview, (xAv, yAv+240), rAv, (0, 255, 0), 4)
				cv2.rectangle(preview, (xAv - 5, yAv+240 - 5), (xAv + 5, yAv+240 + 5), (0, 128, 255), -1)

			elif min_idx ==3:
				#cv2.rectangle(preview, (320,240), (640,480), (0,255,0), 5)


				cv2.circle(preview, (xAv+320, yAv+240), rAv, (0, 255, 0), 4)
				cv2.rectangle(preview, (xAv+320 - 5, yAv+240 - 5), (xAv+320 + 5, yAv+240 + 5), (0, 128, 255), -1)

			elif min_idx ==4:
				#cv2.rectangle(preview, (160,120), (480,360), (0,255,0), 5)

				cv2.circle(preview, (xAv+160, yAv+120), rAv, (0, 255, 0), 4)
				cv2.rectangle(preview, (xAv+160 - 5, yAv+120 - 5), (xAv+160 + 5, yAv+120 + 5), (0, 128, 255), -1)


			#cv2.rectangle(preview, (0,0), (640,480), (0,255,0), 10)
			

		# Estimate Z from diameter
		# size 3 ball is 58-60cm in circumference
		#diameter = Circumference / Pi hence 60/Pi=19.09cm diameter 9.54cm radius
		# radius at 1.5m =
		
		knownWidth = 19.09		
		perWidth = rAv * 2
		focal_length = 494.2584686279297 #for the PS3 EYE

		if rAv !=0:
			distance = (knownWidth * focal_length) / perWidth
			print "Approx XYZ of the ball: %d %d %.2f cm" % (xAv, yAv, distance)
			cv2.putText(preview, ('Approx Distance: %d cm'%distance),(5,10), font, 0.4,(255,255,255),1)
		else:
			return 0

		cv2.imshow('Olly Smith Robot Football AI',preview)
		
		cv2.waitKey(1000)

		
		#cv2.waitKey(0)
		predict_hough = time.time()
		hough_time = predict_hough - predict_start
		print "Time to estimate circle = %.2f" % hough_time
		'''
		'''			End of Hough circles			'''
		print "\n\n"
	else:
		print "no_ball"
		cv2.putText(img, ('No Ball'),(5,75), font, 1.2,(0,0,255),2)
		res = cv2.resize(img, (150, 150))
		cv2.imshow('Object Recognition',res)
		print "\n\n"
	# show the result
	#res = cv2.resize(img, (150, 150))
	#cv2.imshow('Object Recognition',res)
	


print("[INFO] model loaded, press any key to begin...\n")
raw_input() #press enter to begin

print("[INFO] sampling frames from webcam...")
stream = WebcamVideoStream(src=1).start() # start webcam input as a thread

count = 0 # frame count
skip =0 # run the prediction every X frames


# Init
# display the video with a frame counter
frame = stream.read()
preview = cv2.resize(frame, (640, 480))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(preview, ('%d'%count),(5,10), font, 0.4,(255,255,255),1)
cv2.imshow('Olly Smith Robot Football AI',preview)

fpsCount = 0
start = time.time()

# Main
while 1:
	frame = stream.read()
	
	key = cv2.waitKey(1) & 0xFF
	
	
	# display the video with a frame counter
	preview = cv2.resize(frame, (640, 480))
	font = cv2.FONT_HERSHEY_SIMPLEX
	#cv2.putText(preview, ('%d'%count),(5,10), font, 0.4,(255,255,255),1)
	cv2.imshow('Olly Smith Robot Football AI',preview)
	
	if skip == 10:
		# Save current frame to folder
		img = cv2.resize(frame, (150, 150))
		cv2.imwrite(os.path.join(path , 'test.jpg'), img)
		predict()
		skip = 0	
	else:
		skip += 1
		count +=1


	# ESC to quit
	if key == 27:
		end = time.time()
		print "\n\nstart time: %d end time: %d total frames: %d" % (start, end, count)
		totalTime = end - start
		fps = count / totalTime
		print "average fps = %d" % fps
		break
	


cv2.destroyAllWindows()
stream.stop()


