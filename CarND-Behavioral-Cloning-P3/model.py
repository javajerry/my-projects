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
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
matplotlib.use('Agg')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('log', 'driving_log.csv', 'Driving log')

samples = []
steer_angles = []
print('read csv file')

zero_steering_threshold=1400
positive_low_angles = zero_steering_threshold / 2
negative_low_angles = zero_steering_threshold / 2
angle_cutoff = 0.095
positive_steering_count = 0 
negative_steering_count = 0 

with open(FLAGS.log) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
		# collect steering angles
		steer_angles.append(float(line[3]))
		
		#downsample zero steering data
		#if (float(line[3]) >= -angle_cutoff and float(line[3]) <= 0 ) and negative_steering_count <= negative_low_angles and float(line[5]) == 0:
		#	samples.append(line)
		#	negative_steering_count += 1
		#	steer_angles.append(float(line[3]))
		#elif (float(line[3]) > 0 and float(line[3]) <= angle_cutoff ) and positive_steering_count <= positive_low_angles and float(line[5]) == 0:
		#	samples.append(line)
		#	positive_steering_count += 1
		#	steer_angles.append(float(line[3]))
		#elif (float(line[3]) < -angle_cutoff or float(line[3]) > angle_cutoff):
		#	samples.append(line)
		#	steer_angles.append(float(line[3]))


print('Found samples ', len(samples))
#print('Ignored samples ', count)


#split samples into training and validation data, using 20%
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)		

#base path for images

flags.DEFINE_string('image_path','IMG/', "Image path")

#steering correction
correction = 0.25
#image shape
flags.DEFINE_integer('ch', 3, "number of channels")
flags.DEFINE_integer('row', 160, "rows")  
flags.DEFINE_integer('col', 320, "columns")

#CNN
filters=[5,5]

#augment_choice = ['none', 'left', 'right']
augment_choice = ['none']


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

#warp affine
def trans_image(image,steer, trans_range=100):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(col,row))
    
    return image_tr,steer_ang    

# samples generator
def generator(samples, training=False, batch_size=32):
    
	num_samples = len(samples)

	while 1: # Loop forever so the generator never terminates

		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			#print ('offset' , offset)
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []

			for batch_sample in batch_samples :
				#Add center image
				center_source_path = batch_sample[0]
				#print('center', center_source_path)	
				filename = center_source_path.split('\\')[-1]
				#print(filename)
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				#crop the image
				#image = crop(image)
				#print(image.shape)
				images.append(image)

				angle = float(batch_sample[3])
				#print('angle ', angle)
				angles.append(angle)

				images.append(image)
				angles.append(angle)	

				#data augmentation#1
				#image = random_brightness(image)
				#images.append(image)
				#angles.append(angle)		

				#data augmentation#1
				#image = cv2.imread(current_path)
				#image, angle = trans_image(image, angle)
				#images.append(image)
				#angles.append(angle)		

				if training:
					# read image again
					if ( angle < -angle_cutoff or angle  > angle_cutoff ) :
						#image = cv2.imread(current_path)
						image = np.fliplr(image)
						images.append(image)
						angles.append(angle*-1.0)	
						#data augmentation #1	
						#image = random_brightness(image)
						#images.append(image)
						#angles.append(angle*-1.0)		
						#data augmentation#1
						#image = cv2.imread(current_path)
						#image, angle = trans_image(image, angle)
						#images.append(image)
						#angles.append(angle*-1.0)		



				#Add left image
				left_source_path = batch_sample[1]
				filename = left_source_path.split('\\')[-1]
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				#images.append(image)
				#angles.append(angle + correction)

				#Add right image
				right_source_path = batch_sample[2]
				filename = right_source_path.split('\\')[-1]
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				#images.append(image)
				#angles.append(angle - correction)


			x_train = np.array(images)
			#print('x_train shape', x_train.shape)
			y_train = np.array(angles)	
			#print('y_train ', len(y_train))
			yield shuffle(x_train, y_train)


#based on LeNet Architecture
def LeNet_model(ch, row, col):
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(FLAGS.row, FLAGS.col, FLAGS.ch)))
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
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(FLAGS.row, FLAGS.col, FLAGS.ch)))
	#model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(row, col, ch)))
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(FLAGS.row, FLAGS.col, FLAGS.ch)))
	model.add(Convolution2D(24, filters[0], filters[1], subsample=(2,2), activation="relu"))
	#model.add(MaxPooling2D())	
	model.add(Convolution2D(36, filters[0], filters[1], subsample=(2,2), activation="relu"))
	#model.add(MaxPooling2D())
	model.add(Convolution2D(48, filters[0], filters[1], subsample=(2,2), activation="relu"))
	#model.add(BatchNormalization())
	#model.add(MaxPooling2D())
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	#model.add(Convolution2D(64, 3, 3, activation="relu"))	

	model.add(Flatten())
	model.add(Activation('relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(1024)) #100
	#model.add(Activation('relu'))
	#model.add(Dropout(dropout))
	model.add(Dense(512))   #50	
	#model.add(Activation('relu'))
	#model.add(Dropout(dropout))
	model.add(Dense(128))  #10
	#model.add(Activation('relu'))
	model.add(Dense(1))

	return model;

def main(_):
	#check data collection
	plt.figure()
	y1 = np.array(steer_angles)
	h = plt.hist(y1, bins=100)
	s = plt.savefig("hist1.png", format='png', bbox_inches='tight')
	pp.pprint(describe(y1)._asdict())

	#masking small angles to see new variance
	#f = plt.figure()
	#p = lambda x: abs(x)<0.01
	#y2 = np.array([s for s in filterfalse(p,y1)])
	#h = plt.hist(y2,bins=100)
	#s = plt.savefig("hist2.png", format='png', bbox_inches='tight')
	#print("")
	#pp.pprint(describe(y2)._asdict())

	train_generator = generator(train_samples, training=True, batch_size=64)
	validation_generator = generator(validation_samples, batch_size=64)
	#model = LeNet_model(ch, row, col)
	model = nvidia_model(FLAGS.ch, FLAGS.row, FLAGS.col);

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
	plt.figure()
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	#plt.show()
	s = plt.savefig("mse_accu.png", format='png', bbox_inches='tight')

if __name__ == '__main__':
    tf.app.run()
