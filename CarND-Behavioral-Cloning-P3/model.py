import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

samples = []
print('read csv file')

with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)		

image_path = 'IMG/'

#crop images
def crop(image):
    return image[50:-20,:]

# samples generator
def generator(samples, batch_size=32):
    
	num_samples = len(samples)

	while 1: # Loop forever so the generator never terminates

		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

		images = []
		angles = []

		for batch_sample in batch_samples :
			#Add center image
			center_source_path = batch_sample[0]
			#print(source_path)	
			filename = center_source_path.split('\\')[-1]
			#print(filename)
			current_path = image_path + filename
			image = cv2.imread(current_path)
			#crop the image
			image = crop(image)
			#print(image.shape)
			images.append(image)

			angle = float(batch_sample[3])
			angles.append(angle)

			#Add flipped image also
			image = np.fliplr(image)
			images.append(image)
			angles.append(angle)			
	
			#Add left image
			left_source_path = batch_sample[1]
			#print(source_path)	
			filename = left_source_path.split('\\')[-1]
			#print(filename)
			current_path = image_path + filename
			image = cv2.imread(current_path)
			#crop the image
			image = crop(image)
			images.append(image)

			angles.append(angle)
			
			#Add flipped image also
			image = np.fliplr(image)
			images.append(image)
			angles.append(angle)			

			#Add right image
			right_source_path = batch_sample[2]
			#print(source_path)	
			filename = right_source_path.split('\\')[-1]
			#print(filename)
			current_path = image_path + filename
			image = cv2.imread(current_path)
			#crop the image
			image = crop(image)
			images.append(image)

			angles.append(angle)

			#Add flipped image also
			image = np.fliplr(image)
			images.append(image)
			angles.append(angle)			

		x_train = np.array(images)
		print('x_train shape', x_train.shape)
		y_train = np.array(angles)	
		yield shuffle(x_train, y_train)

def LeNet_model(ch, row, col):
	model = Sequential()
	#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
	model.add(Convolution2D(6, 5, 5, activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6, 5, 5, activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model;


def main(_):
	train_generator = generator(train_samples, batch_size=64)
	validation_generator = generator(validation_samples, batch_size=64)

	ch, row, col = 3, 90, 320  # Trimmed image format

	model = LeNet_model(ch, row, col);
	
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	#model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)
	model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

	model.save('model.h5')

if __name__ == '__main__':
    tf.app.run()
