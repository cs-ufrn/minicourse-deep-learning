# Import sklearn and keras tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np


# Prepare the feedforward neural network with keras
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

dimData = np.prod(trainX.shape[1:])
trainX = trainX.reshape(trainX.shape[0], dimData)
testX = testX.reshape(testX.shape[0], dimData)


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
width = 32
height = 32
depth = 3
classes = 10
model = Sequential()
inputShape = (height, width,depth)

model.add(Dense(units=256, input_shape=(3072,) ,activation="sigmoid"))
model.add(Dense(units=128, activation="sigmoid"))
model.add(Dense(units=10, activation="softmax"))

# Train the neural network
print("[INFO] training...")
sgd = SGD(0.1)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128, verbose=1)

model.save('ff.h5')
