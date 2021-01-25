import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout , BatchNormalization
from keras.callbacks import ReduceLROnPlateau

train_df = pd.read_csv("csv_data/train_data.csv")
test_df = pd.read_csv("csv_data/test_data.csv")

# number of keypoints per hand
kp_num = 21

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
model.add(Dense(units = kp_num * 2 , activation = 'relu'))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 25 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

t = time.time()

history = model.fit(x_train,y_train, batch_size = 256 ,epochs = 200 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

print("training time:", time.time() - t, "s")

model.save("models/model_4")

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")