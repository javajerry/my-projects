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
from sklearn.model_selection import train_test_split
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
flags.DEFINE_integer('threshold', 300, 'limiting number of images')
flags.DEFINE_float('angle_cutoff', 0.1, "Used for Image flipping")


#base path for images

flags.DEFINE_string('image_path','IMG/', "Image path")

#steering correction
flags.DEFINE_float('correction', 0.25, "Angle offset for Steering") 
#image shape
flags.DEFINE_integer('ch', 3, "number of channels")
flags.DEFINE_integer('row', 160, "rows")  
flags.DEFINE_integer('col', 320, "columns")

#CNN
filters=[3,3]
# for fully connected
dense_params_set1 = [100,50,10,1]
dense_params_set2 = [1024,512,128,1]


samples = []
steer_angles = []

'''
	Read driving_log.csv

'''
def read_log():
	global samples, steer_angles
	with open(FLAGS.log) as csvfile:
		print('read csv file')
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
			steer_angles.append(float(line[3]))
		print('Found samples ', len(samples))

'''
	Data cleanup. Original training dat is very imbalanced. lot of data collected
	with steer angle = 0 plus minus 0.05 degrees. need to reduce the data and randomly choose
	data for those angles
'''
def data_cleanup():
	global samples, steer_angles
	angles_list = np.unique(np.array(steer_angles))
	#print ( angles_list)
	filter_angles = []
	nonfilter_angles = []

	for ang in angles_list:
	    locs = np.where(steer_angles == ang)
	    #print(ang, '  ', len(locs[0]))
	    if len(locs[0]) > FLAGS.threshold:
	        filter_angles.append(ang)
	    else:
	        nonfilter_angles.append(ang)

	print(filter_angles)
	#print(nonfilter_angles)

	nonfilter_angles_set = set(nonfilter_angles)

	clean_samples = []
	clean_steer_angles = []

	for sample in samples:
	    #print ( sample[3])
	    if float(sample[3]) in nonfilter_angles_set:
	        clean_samples.append(sample)
	        clean_steer_angles.append(float(sample[3]))

	samples_array = np.array(samples)        
	for ang in filter_angles:
	    locs = np.where(steer_angles == ang)
	    
	    for i in range(FLAGS.threshold) :
	        loc = random.choice(locs[0])
	        #print (samples_array[loc])
	        clean_samples.append(samples_array[loc])
	        clean_steer_angles.append(float(samples_array[loc][3]))

	#plt.figure()
	#y1 = np.array(clean_steer_angles)
	#h = plt.hist(y1, bins=100)
	#plt.show()

	#clean_angles_list = np.unique(np.array(clean_steer_angles))

	#for ang in clean_angles_list:
	#    locs = np.where(clean_steer_angles == ang)
	#    print(ang, '  ', len(locs[0]))

	# reset samples and steer_angles
	samples = clean_samples
	steer_angles = clean_steer_angles


'''
If same angle is captured more than 5 times, most like its same image.
Ignore the extra images
'''
def data_cleanup2():
	global samples, steer_angles
	last5_angles = list()
	clean_samples = []
	clean_steer_angles = []

	for sample in samples:
		if len(last5_angles) < 5:
			clean_samples.append(sample)
			clean_steer_angles.append(float(sample[3]))
			last5_angles.append(sample[3])
		else:
			angle = sample[3]
			#print ( last5_angles.count(angle) )
			if last5_angles.count(angle) < 5:
				last5_angles.pop(0)
				clean_samples.append(sample)
				clean_steer_angles.append(float(sample[3]))
				last5_angles.append(sample[3])

	# reset samples and steer_angles
	samples = clean_samples
	steer_angles = clean_steer_angles

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

#warp affine only
def trans_image(image,steer, trans_range=100):

	image_shape = np.shape(image)
	# Translation
	tr_x = trans_range*np.random.uniform()-trans_range/2
	steer_ang = steer + tr_x/trans_range*2*.2
	tr_y = 40*np.random.uniform()-40/2
	#tr_y = 0
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	image_tr = cv2.warpAffine(image,Trans_M,(image_shape[1],image_shape[0]))
    
	return image_tr,steer_ang    

'''
	Based on Nvidia CNN document, doing randome rotation and warp affine
'''
def transform_image(x, rot_angle=0.05, shift=0.09):
    image_shape = np.shape(x)
    rand_angle = np.random.uniform(rot_angle)-rot_angle/2.
    rand_shift_x = shift*np.random.uniform()-shift/2. 
    rand_shift_y = shift*np.random.uniform()-shift/2.
    
    rotate = cv2.getRotationMatrix2D((image_shape[1]/2,image_shape[0]/2),rand_angle,1.)
    trans = np.float32([[1,0,rand_shift_x],[0,1,rand_shift_y]])
        
    x = cv2.warpAffine(x,rotate,(image_shape[1],image_shape[0]))
    x = cv2.warpAffine(x,trans,(image_shape[1],image_shape[0]))
    return x    	

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

				if random.random() > 0.5:
					#image = cv2.imread(current_path)
					image = np.fliplr(image)
					images.append(image)
					angles.append(angle*-1.0)	

				#data augmentation#1 :
				if random.random() > 0.5:
					image = cv2.imread(current_path)
					image = random_brightness(image)
					images.append(image)
					angles.append(angle)		

				#data augmentation#1
				if random.random() > 0.5:
					image = cv2.imread(current_path)
					image, angle = trans_image(image, angle)
					images.append(image)
					angles.append(angle)		


				#Add left image
				left_source_path = batch_sample[1]
				filename = left_source_path.split('\\')[-1]
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				images.append(image)
				angle_left = angle + FLAGS.correction
				angles.append(angle_left)

				if random.random() > 0.5:
					#image = cv2.imread(current_path)
					image = np.fliplr(image)
					images.append(image)
					angles.append(angle_left * -1.0)	

				#data augmentation#1 :
				if random.random() > 0.5:
					image = cv2.imread(current_path)
					image = random_brightness(image)
					images.append(image)
					angles.append(angle_left)		

				#data augmentation#1
				if random.random() > 0.5:
					image = cv2.imread(current_path)
					image, angle_left = trans_image(image, angle_left)
					images.append(image)
					angles.append(angle_left)		

				#Add right image
				right_source_path = batch_sample[2]
				filename = right_source_path.split('\\')[-1]
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				images.append(image)
				angle_right = angle - FLAGS.correction
				angles.append(angle_right)

				if random.random() > 0.5:
					#image = cv2.imread(current_path)
					image = np.fliplr(image)
					images.append(image)
					angles.append(angle_right * -1.0)	

				#data augmentation#1 :
				if random.random() > 0.5:
					image = cv2.imread(current_path)
					image = random_brightness(image)
					images.append(image)
					angles.append(angle_right)		

				#data augmentation#1
				if random.random() > 0.5:
					image = cv2.imread(current_path)
					image, angle_right = trans_image(image, angle_right)
					images.append(image)
					angles.append(angle_right)		


			x_train = np.array(images)
			#print('x_train shape', x_train.shape)
			y_train = np.array(angles)	
			#print('y_train ', len(y_train))
			yield shuffle(x_train, y_train)

# samples generator
def generator2(samples, training=False, batch_size=32):
    
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

				#data augmentation#1
				if random.random() > 0.5:
					#image = cv2.imread(current_path)
					image= transform_image(image)
					images.append(image)
					angles.append(angle)		


				#Add left image
				left_source_path = batch_sample[1]
				filename = left_source_path.split('\\')[-1]
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				images.append(image)
				angle_left = angle + FLAGS.correction
				angles.append(angle_left)

				if random.random() > 0.5:
					#image = cv2.imread(current_path)
					image= transform_image(image)
					images.append(image)
					angles.append(angle_left * -1.0)	

				#Add right image
				right_source_path = batch_sample[2]
				filename = right_source_path.split('\\')[-1]
				current_path = FLAGS.image_path + filename
				image = cv2.imread(current_path)
				images.append(image)
				angle_right = angle - FLAGS.correction
				angles.append(angle_right)

				if random.random() > 0.5:
					#image = cv2.imread(current_path)
					image= transform_image(image)
					images.append(image)
					angles.append(angle_right * -1.0)	


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
	#model.add(Dropout(0.5))
	#model.add(BatchNormalization())
	#model.add(MaxPooling2D())
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))	

	model.add(Flatten())
	#model.add(Activation('relu'))
	model.add(Dense(1164))
	model.add(Dropout(0.5))
	model.add(Dense(dense_params_set1[0])) #100
	#model.add(Activation('relu'))
	model.add(Dropout(0.5))	
	model.add(Dense(dense_params_set1[1]))   #50	
	model.add(Dropout(0.5))
	#model.add(Activation('relu'))
	#model.add(Dropout(dropout))
	model.add(Dense(dense_params_set1[2]))  #10
	#model.add(Activation('relu'))
	model.add(Dense(dense_params_set1[3]))

	return model;

def main(_):

	read_log()

	#check data collection
	plt.figure()
	y1 = np.array(steer_angles)
	h = plt.hist(y1, bins=100)
	s = plt.savefig("hist1.png", format='png', bbox_inches='tight')
	pp.pprint(describe(y1)._asdict())

	# balanncing the data to reduce excees data
	data_cleanup2()

	#check data collection
	plt.figure()
	y1 = np.array(steer_angles)
	h = plt.hist(y1, bins=100)
	s = plt.savefig("hist2.png", format='png', bbox_inches='tight')
	pp.pprint(describe(y1)._asdict())


	#split samples into training and validation data, using 20%
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)		


	train_generator = generator2(train_samples, training=True, batch_size=64)
	validation_generator = generator2(validation_samples, batch_size=64)
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
