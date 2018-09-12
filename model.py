import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

samples = []
with open('C:/Udacity/CarND-Behavioral-Cloning-P3-master/data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
del(samples[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				path_center = 'C:/Udacity/CarND-Behavioral-Cloning-P3-master/data/'+batch_sample[0]#.split('/')[-1]
				path_left = 'C:/Udacity/CarND-Behavioral-Cloning-P3-master/data/'+batch_sample[1][1:]
				path_right = 'C:/Udacity/CarND-Behavioral-Cloning-P3-master/data/'+batch_sample[2][1:]
				
				# The lines below read, crop, change the color and resize the left, center and right images.
				# These transformations are also applied in drive.py file
				# Test with YUV color space.
				center_image = cv2.imread(path_center)
				center_image = center_image[25:90,:,:]			
				center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
				center_image = cv2.resize(center_image,(200, 70), interpolation = cv2.INTER_AREA)
				
				left_image = cv2.imread(path_left)
				left_image = left_image[25:90,:,:]
				left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
				left_image = cv2.resize(left_image,(200, 70), interpolation = cv2.INTER_AREA)
				
				right_image = cv2.imread(path_right)
				right_image = right_image[25:90,:,:]
				right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
				right_image = cv2.resize(right_image,(200, 70), interpolation = cv2.INTER_AREA)
				
				#if center_image is None:
					#print(path_center)
					
				# The lines below determines the correct angle for left, center and right images 
				# and affects them to "images" and "angles" lists.
				center_angle = float(batch_sample[3])
				correction = 0.2
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				
				images.append(center_image)
				images.append(left_image)
				images.append(right_image)
				
				angles.append(center_angle)
				angles.append(left_angle)
				angles.append(right_angle)
				
				# Take more than 0.5 for angle value avoids to overfit with very similar data.
				if abs(center_angle) > 0.5: 
					center_image_flip = cv2.flip(center_image, 1)
					left_image_flip = cv2.flip(left_image, 1)
					right_image_flip = cv2.flip(right_image, 1)
					
					center_angle_flip = center_angle * -1.0
					left_angle_flip = right_angle * -1.0
					right_angle_flip = left_angle * -1.0
					
					images.append(center_image_flip)
					images.append(left_image_flip)
					images.append(right_image_flip)
					
					angles.append(center_angle_flip)
					angles.append(left_angle_flip)
					angles.append(right_angle_flip)
					

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


ch, row, col = 3, 70, 200  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint

# Below is the convolutional neural network.

model = Sequential()


# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row,col,ch)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath='/tmp/weights.h5', verbose=1, save_best_only=True)
history_object = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=9)
			
model.save('model.h5')

### plot the training and validation loss for each epoch
plt.figure(2)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()