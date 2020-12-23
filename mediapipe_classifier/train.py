import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

y_train = train_df['letter']
y_test = test_df['letter']
del(train_df['letter'])
del(test_df['letter'])

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

model = Sequential()
model.add(Dense(units = 42 , activation = 'relu'))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 25 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

history = model.fit(x_train,y_train, batch_size = 256 ,epochs = 100 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

model.save("model_1")

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")