'''
Olly Smith 2018
olly.smith1994@gmail.com

The program uses augmentation (ImageDataGenerator) to train a CNN model to perform object recognition on the presence of a size 3 football in an image.
'''

print'ball vs no ball using CNN w/ image augmentation\n'

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import time # for recording time for training
start = time.time()
#from keras import backend as K


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 43468+45042
nb_validation_samples = 10866+11260
epochs = 10
batch_size = 32


# added 14.5.18
learning_rate = 0.1
decay_rate = learning_rate / epochs

# EPOCHS SET TO 1, CONFIRM CODE WORKS.

#Don't think I need this as I have the keras.json file set to channels first?
'''
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
'''

# input_shape to channels first
input_shape = (3, img_width, img_height)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# checkpoints (will save if there's an improvement after the epoch)
from keras.callbacks import ModelCheckpoint
filepath="weights-best-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
callbacks_list = [checkpoint]


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	rotation_range=30, # 30 as the application shouldn't feature much rotation of camera
	width_shift_range=0.2, # vary position of object in frame
	height_shift_range=0.2,
	)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

history = model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	callbacks=callbacks_list,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)

#	SAVING THE MODEL
# SAVE
print("[INFO] saving model...\n")
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("[INFO] model saved.\n")

end = time.time()
totalTime = end - start
print "Total time to train dataset: %d" % totalTime
mins = totalTime / 60
print "AKA: %.2f minutes" % (mins)

# New as of 14.5.18
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








