import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images=[]
measurements = []
image_path = 'IMG/'

print('read csv file')
for line in lines:
	#Add center image
	center_source_path = line[0]
	#print(source_path)	
	filename = center_source_path.split('\\')[-1]
	#print(filename)
	current_path = image_path + filename
	image = cv2.imread(current_path)
	images.append(image)

	#Add left image
	left_source_path = line[1]
	#print(source_path)	
	filename = left_source_path.split('\\')[-1]
	#print(filename)
	current_path = image_path + filename
	image = cv2.imread(current_path)
	images.append(image)

	#Add right image
	right_source_path = line[2]
	#print(source_path)	
	filename = right_source_path.split('\\')[-1]
	#print(filename)
	current_path = image_path + filename
	image = cv2.imread(current_path)
	images.append(image)


	measurement = float(line[3])
	measurements.append(measurement)

x_train = np.array(images)
print('x_train shape', x_train.shape)
y_train = np.array(measurements)	

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

model.save('model.h5')