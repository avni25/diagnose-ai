import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tkinter import image_names
import cv2 as cv
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.utils import to_categorical


dataset = []
label =[]
INPUT_SIZE = 64
img_directory = "dataset/"

pics_no_tumor = os.listdir(img_directory + "no/")
pics_yes_tumor = os.listdir(img_directory + "yes/")


for i, img_name in enumerate(pics_no_tumor):
    if(img_name.split(".")[1] == "jpg"):
        img = cv.imread(img_directory + "/no/" + img_name)
        img_arr = Image.fromarray(img, "RGB")
        resized_img = img_arr.resize((64,64))
        dataset.append(np.array(resized_img))
        label.append(0)

for i, img_name in enumerate(pics_yes_tumor):
    if(img_name.split(".")[1] == "jpg"):
        img = cv.imread(img_directory + "/yes/" + img_name)
        img_arr = Image.fromarray(img, "RGB")
        resized_img = img_arr.resize((64,64))
        dataset.append(np.array(resized_img))
        label.append(1)

ds = np.array(dataset)
lbl = np.array(label)



x_train, x_test, y_train, y_test =  train_test_split(ds, lbl, test_size=0.2, random_state=0)

# reshape = (n, width, height, n_channel)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)


# Model Building

model= Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform" ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, 
epochs=10, 
validation_data=(x_test, y_test),
shuffle=False)

model.save("dianseAIepochs10.h5")











cv.waitKey(0)
