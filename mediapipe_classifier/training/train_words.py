import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout , BatchNormalization
from keras.callbacks import ReduceLROnPlateau

train_df = pd.read_csv("csv_data/train_data_words.csv")
test_df = pd.read_csv("csv_data/test_data_words.csv")

# number of keypoints per hand
kp_num = 21

y_train = train_df['word']
y_test = test_df['word']
del(train_df['word'])
del(test_df['word'])

words = ['name', 'i', 'hello', 'learn', 'yes', 'no', 'brother', 'you']
from sklearn.preprocessing import label_binarize
# label_binarizer = LabelBinarizer()
y_train = label_binarize(y_train, classes=words)
y_test = label_binarize(y_test, classes=words)
print(y_train)
print(y_test)

x_train = train_df.values
x_test = test_df.values

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

# input_size = 2 (for each hand) * (number of keypoints per hand) * 2 (for x & y of each keypoint) + 2 (x & y coordinates of the face)
input_size = 4 * kp_num + 2

model = Sequential()
model.add(Dense(units = input_size, activation = 'relu'))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 8 , activation = 'softmax'))
model.compile(optimizer = 'sgd' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

t = time.time()

history = model.fit(x_train,y_train, batch_size = 128 ,epochs = 200 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

print("training time:", time.time() - t, "s")

model.save("models_words/model_0")

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")