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

model = keras.models.load_model("models_words/model_0")

import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# number of keypoints per hand
kp_num = 21

hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
count = 0
t1 = time.time()
words = ['name', 'i', 'hello', 'learn', 'yes', 'no', 'brother', 'you']

face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')

while cap.isOpened():
    count += 1
    success, image = cap.read()
    if not success:
        break
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        row = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.landmark:
                row += [i.x, i.y]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if len(row) < kp_num * 4:
            row += [0] * (kp_num * 4 - len(row))
        if len(faces) > 0:
            row += [(faces[0, 0] + faces[0, 2]//2)/1280, (faces[0, 1] + faces[0, 3]//2)/720]
        else:
            row += [0.5, 0.5]
        row = np.array(row).reshape(1,kp_num * 4 + 2)
        prediction = model.predict(row)
        guess = np.argmax(prediction)
        cv2.putText(image,words[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        
    cv2.imshow('MediaPipe Hands', image)
    #print(count/(time.time() - t1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
  
# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 