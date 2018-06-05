'''
Olly Smith 2018
olly.smith1994@gmail.com

This program uses videos of data, stored in folders, to generate an image dataset for training a model. The folders contain videos of data, by class (ball, no_ball). The program will go through each video, and save frames to a folder of that class.


16.4.18
Preparing data for what will hopefully be the last time, barring segmentation implementation.

This works brill, adding named windows would be nice, as well as removing the repeated code.
'''

import cv2
import os
import glob

# INPUT FOLDERS
Ball_folder =		"/media/olly/Storage/Uni/project/may_demo/training_vids/ball_vids/*"
No_Ball_folder =	"/media/olly/Storage/Uni/project/may_demo/training_vids/no_ball_vids/*"

# OUTPUT FOLDERS
train_ball =	"/home/olly/Desktop/may_demo/dataset_5/data/train/ball"
train_no_ball =	"/home/olly/Desktop/may_demo/dataset_5/data/train/no_ball"
test_ball =	"/home/olly/Desktop/may_demo/dataset_5/data/test/ball"
test_no_ball =	"/home/olly/Desktop/may_demo/dataset_5/data/test/no_ball"


SKIP = 4 # every Nth+1 frame to use 

SPLIT = 4 # This number determines the train-test split ratio. Eg 4, means for every 4 train images, there will be a test image, ergo a 80/20 train-test split. 3 = 75/25



frameskip = 0	# choosing to save every X frame of the video
train_test = 0	# the ratio of frames for training vs testing
trainCount = 0	# used for saving filenames
testCount = 0	# used for saving filenames
	


# For the no_ball data
for name in glob.glob(Ball_folder): #'dir/*'
	print name

	cap = cv2.VideoCapture(name)
	


	while True :
		ret, frame = cap.read()
		if ret == True:
			
			# ESC to stop
			k = cv2.waitKey(1)
			if k==27:
				cv2.destroyAllWindows()
				break
	
			# Preview the image
			preview = cv2.resize(frame, (480, 270))
			#cv2.imshow("preview", preview)
			
			# Lets prepare our dataset
			if frameskip == SKIP: # every Nth+1 frame
				cv2.waitKey(1)


				if train_test == SPLIT:
					# resize image to network input size		
					test = cv2.resize(frame, (150, 150))
					#cv2.imshow("save test", test)
					# Save the frame, inc count
					cv2.imwrite(os.path.join(test_ball,("ball_%d.jpg" % testCount)), test)
					print "saved test ball_%d" % testCount
					testCount +=1
					train_test = 0 # reset train_test count
					frameskip = 0

				train = cv2.resize(frame, (150, 150))
				#cv2.imshow("save train", train)
				# Save the frame, inc count
				cv2.imwrite(os.path.join(train_ball,("ball_%d.jpg" %trainCount)), train)
				print "saved train ball_%d" % trainCount
				trainCount +=1
				frameskip = 0
				train_test +=1



			frameskip +=1
		else:
			cap.release()
			cv2.destroyAllWindows()
			break

cap.release()
cv2.destroyAllWindows()


'''	Doing exactly the same thing again because it works and i'm lazy		'''

frameskip = 0	# choosing to save every X frame of the video
train_test = 0	# the ratio of frames for training vs testing
trainCount = 0	# used for saving filenames
testCount = 0	# used for saving filenames


# For the ball data
for name in glob.glob(No_Ball_folder): #'dir/*'
	print name

	cap = cv2.VideoCapture(name)
	


	while True :
		ret, frame = cap.read()
		if ret == True:
			
			# ESC to stop
			k = cv2.waitKey(1)
			if k==27:
				cv2.destroyAllWindows()
				break
	
			# Preview the image
			preview = cv2.resize(frame, (480, 270))
			#cv2.imshow("preview", preview)
			
			# Lets prepare our dataset
			if frameskip == SKIP: # every Nth+1 frame
				cv2.waitKey(1)


				if train_test == SPLIT:
					# resize image to network input size		
					test = cv2.resize(frame, (150, 150))
					#cv2.imshow("save test", test)
					# Save the frame, inc count
					cv2.imwrite(os.path.join(test_no_ball,("no_ball_%d.jpg" % testCount)), test)
					print "saved test no_ball_%d" % testCount
					testCount +=1
					train_test = 0 # reset train_test count
					frameskip = 0

				train = cv2.resize(frame, (150, 150))
				#cv2.imshow("save train", train)
				# Save the frame, inc count
				cv2.imwrite(os.path.join(train_no_ball,("no_ball_%d.jpg" %trainCount)), train)
				print "saved train no_ball_%d" % trainCount
				trainCount +=1
				frameskip = 0
				train_test +=1



			frameskip +=1
		else:
			cap.release()
			cv2.destroyAllWindows()
			break

cap.release()
cv2.destroyAllWindows()









