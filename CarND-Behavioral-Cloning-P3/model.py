import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images=[]
measurements = []
for line in lines:
	source_path = line[0]
	print(source_path)	
	filename = source_path.split('\\')[-1]
	print(filename)
	current_path = 'IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

x_train = np.array(images)
print('x_train shape', x_train.shape)
y_train = np.array(measurements)	

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')