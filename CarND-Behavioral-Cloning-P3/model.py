import os
import csv
import cv2
import numpy as np
import random
import pprint as pp
from scipy.stats import kurtosis, skew, describe
from itertools import filterfalse
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
matplotlib.use('Agg')

samples = []
steer_angles = []
print('read csv file')

zero_steering_threshold=1000
zero_steering_count = 0 
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
		# collect steering angles
		steer_angles.append(float(line[3]))
		#downsample zero steering data
		#if (float(line[3]) > 0.03 and float(line[3]) < -0.03) and zero_steering_count <= zero_steering_threshold:
		#	samples.append(line)
		#	zero_steering_count += 1
		#else:
		#	samples.append(line)
print('Found samples ', len(samples))
#print('Ignored samples ', count)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)		

image_path = 'IMG/'
correction = 0.2
ch, row, col = 3, 160, 320  # Trimmed image format


filters=[3,3]

center_image_choices = ['Brightness', 'WrapAffine']
lr_image_choices = ['None', 'Flip', 'Brightness', 'WrapAffine']

#crop images
def crop(image):
    return image[50:-20,:]

'''
For data augmentation, random brightness
'''
def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#wrap affine
def trans_image(image,steer, trans_range=100):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(col,row))
    
    return image_tr,steer_ang

def augment_center_image(image, angle):
	choice = random.choice(center_image_choices)

	if choice == 'Brightness':
		image = random_brightness(image)
	elif choice == 'WrapAffine':
		image, angle = trans_image(image, angle)
	return image, angle

def augment_lr_image(image, angle):
	choice = random.choice(center_image_choices)

	if choice == 'Brightness':
		image = random_brightness(image)
	elif choice == 'WrapAffine':
		image, angle = trans_image(image, angle)
	elif choice == 'Flip':
		image = np.fliplr(image) 
		angle = -angle
	return image, angle

# samples generator
def generator(samples, training=false, batch_size=32):
    
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
				#print('center', center_source_path)	
				filename = center_source_path.split('\\')[-1]
				#print(filename)
				current_path = image_path + filename
				image = cv2.imread(current_path)
				#crop the image
				#image = crop(image)
				#print(image.shape)
				images.append(image)

				angle = float(batch_sample[3])
				#print('angle ', angle)
				angles.append(angle)

				image, angle = augment_center_image(image, angle)
				images.append(image)
				angles.append(angle)			
		
				#Add left image
				left_source_path = batch_sample[1]
				#print('left ' , left_source_path)	
				filename = left_source_path.split('\\')[-1]
				#print(filename)
				current_path = image_path + filename
				#print(current_path)
				image = cv2.imread(current_path)
				#crop the image
				#image = crop(image)
				images.append(image)

				angles.append(angle + correction)
				
				#Add flipped image also
				image, angle = augment_lr_image(image, angle)
				images.append(image)
				angles.append((angle + correction))			

				#Add right image
				right_source_path = batch_sample[2]
				#print('right ' , right_source_path)	
				filename = right_source_path.split('\\')[-1]
				#print(filename)
				current_path = image_path + filename
				image = cv2.imread(current_path)
				#crop the image
				#image = crop(image)
				images.append(image)

				angles.append(angle - correction)

				#Add flipped image also
				image, angle = augment_lr_image(image, angle)
				images.append(image)
				angles.append((angle - correction))			

		x_train = np.array(images)
		#print('x_train shape', x_train.shape)
		y_train = np.array(angles)	
		#print('y_train ', len(y_train))
		yield shuffle(x_train, y_train)

def generator2(samples, batch_size=32):
    
	num_samples = len(samples)
    
	while 1: # Loop forever so the generator never terminates

		images = []
		angles = []

		for batch_idx in range(batch_size):
			#pick any random image
			rand_idx = random.randint(num_samples)
			#pick any random center 0,left 1, right 2 image
			rand_img = random.randint(2)

			offset = 0


			if rand_img == 1 : #left 
				source_path = batch_sample[rand_img]
				filename = source_path.split('\\')[-1]
				#print(filename)
				current_path = image_path + filename
				offset = 0.2
			elif rand_img == 2 : #right 
				source_path = batch_sample[rand_img]
				filename = source_path.split('\\')[-1]
				#print(filename)
				current_path = image_path + filename
				offset = -0.2
			elif rand_img == 0 : #center 
				source_path = batch_sample[rand_img]
				filename = source_path.split('\\')[-1]
				#print(filename)
				current_path = image_path + filename

			image = cv2.imread(current_path)
			angle = float(batch_sample[3] + offset)

			image, angle = augment_lr_image(image, angle)
			images.append(image)
			angles.append(angle)			
		
		x_train = np.array(images)
		#print('x_train shape', x_train.shape)
		y_train = np.array(angles)	
		#print('y_train ', len(y_train))
		yield shuffle(x_train, y_train)

#based on LeNet Architecture
def LeNet_model(ch, row, col):
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Convolution2D(6, 5, 5, activation="relu"))
	model.add(MaxPooling2D())
	#model.add(Dropout(0.5))
	model.add(Convolution2D(6, 5, 5, activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model;

def nvidia_model(ch, row, col, dropout=0.4):
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
	#model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(row, col, ch)))
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Convolution2D(24, filters[0], filters[1], subsample=(2,2), activation="relu"))
	#model.add(MaxPooling2D())	
	model.add(Convolution2D(36, filters[0], filters[1], subsample=(2,2), activation="relu"))
	#model.add(MaxPooling2D())
	model.add(Convolution2D(48, filters[0], filters[1], subsample=(2,2), activation="relu"))
	#model.add(MaxPooling2D())
	#model.add(Convolution2D(64, 3, 3, activation="relu"))
	#model.add(Convolution2D(64, 3, 3, activation="relu"))	

	model.add(Flatten())
	model.add(Dense(100))
	#model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(50))	
	#model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(10))
	model.add(Dense(1))
	return model;

def main(_):
	#check data collection
	y1 = np.array(steer_angles)
	h = plt.hist(y1, bins=100)
	s = plt.savefig("hist1.png", format='png', bbox_inches='tight')
	pp.pprint(describe(y1)._asdict())

	#masking small angles
	f = plt.figure()
	p = lambda x: abs(x)<0.01
	y2 = np.array([s for s in filterfalse(p,y1)])
	h = plt.hist(y2,bins=100)
	s = plt.savefig("hist2.png", format='png', bbox_inches='tight')
	print("")
	pp.pprint(describe(y2)._asdict())

	train_generator = generator(train_samples, batch_size=128)
	validation_generator = generator(validation_samples, batch_size=128)

	#model = LeNet_model(ch, row, col);
	model = nvidia_model(ch, row, col);

	model.summary()
	plot(model, to_file='model.png', show_shapes=True)
	
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	#model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)
	early_stopping = EarlyStopping(monitor='val_loss', patience=2)
	history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, callbacks=[early_stopping], verbose=1)

	model.save('model.h5')

	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

if __name__ == '__main__':
    tf.app.run()
