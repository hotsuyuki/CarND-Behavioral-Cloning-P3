import csv
import cv2
import numpy as np

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    # 0:center, 1:left, 2:right
    for direction in range(3):
        # directory of the picture 
        source_path = line[direction]
        filename = source_path.split('/')[-1]
        current_path = './IMG/' + filename

        # read the images
        image = cv2.imread(current_path)
        images.append(image)
        correction = 0.1
        if (direction == 0): # center
            measurement = float(line[3])
        elif (direction == 1): # left
            measurement = float(line[3])-correction
        elif (direction == 2): # right
            measurement = float(line[3])+correction
        measurements.append(measurement)

        # flip the images
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,28,28, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(6,10,10, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

