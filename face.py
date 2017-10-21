## For reproduction
import numpy as np
np.random.seed(100)

## Data processing
from keras.utils import np_utils
len_data = 10000
data = open("face.csv", "r").readlines()[1:]
xtrain  = []
ytrain  = []

for i in range(len_data):
	line = data[i].split(",")
	ytrain.append(float(line[0])*1.0)
	line[1] = line[1].split(" ")

	tmp = []
	pixels = []
	for i in range(len(line[1])):
		if i%48 == 0 and i != 0:
			pixels.append(tmp)
			tmp = []
		tmp.append(float(line[1][i])*1.0)
	pixels.append(tmp)
	pixels = np.array(pixels)
	xtrain.append(pixels)

del data
xtrain = np.array(xtrain)
xtrain /= 255.0
ytrain = np_utils.to_categorical(ytrain, 7)
xtrain = xtrain.reshape(len_data, 1, 48, 48)

## model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
import keras.backend as K
K.set_image_dim_ordering("th")

model = Sequential()
model.add(Convolution2D(32,3,3,activation="relu",input_shape=(1,48,48)))
model.add(Convolution2D(32,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32,3,3,activation="relu"))
model.add(Convolution2D(32,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(212, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(xtrain, ytrain, batch_size=200, nb_epoch=10)

score = model.evaluate(xtrain, ytrain)
print(score)
