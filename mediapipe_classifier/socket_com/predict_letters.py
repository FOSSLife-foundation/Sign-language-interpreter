import numpy as np
import keras
model = keras.models.load_model("../models/model_4")

import cv2
import imutils
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.5)

# number of keypoints per hand
kp_num = 21

count = 0
t1 = time.time()
letters = 'abcdefghijklmnopqrstuvwxy'

def process_image(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row = []
            for i in hand_landmarks.landmark:
                row += [i.x, i.y]
            row = np.array(row).reshape(1, kp_num * 2)
            prediction = model.predict(row)
            guess = np.argmax(prediction)
            cv2.putText(image,letters[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image

hands.close()
cap.release()
cv2.destroyAllWindows() 