import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy") and not(i.startswith("labels")):  
        data = np.load(i)
        if not is_init:
            is_init = True 
            X = data
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

y = to_categorical(y)

# Shuffling the data
shuffle_indices = np.arange(X.shape[0])
np.random.shuffle(shuffle_indices)

X = X[shuffle_indices]
y = y[shuffle_indices]

# Building the model
input_layer = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(input_layer)
m = Dense(256, activation="relu")(m)
output_layer = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)

# Saving the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))

