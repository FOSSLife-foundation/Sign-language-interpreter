import mediapipe as mp
import cv2
import numpy as np
import os
import keras

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.5)

kp_num = 21
model_letters = keras.models.load_model("C:/Users/ASUS/Desktop/Automation Challenge/Sign-language-interpreter/ASLInterpreter/models/model_letters")
model_words = None
letters =  ['_switch'] + list('abcdefghijklmnopqrstuvwxy')
words = ['_switch', 'name', 'i', 'hello', 'learn', 'yes', 'no', 'thank', 'you', 'who', 'what', 'when', 'where', 'why', 'how', 'which']

def load_letters_model(path):
    """
    Function to load letter detection model from path.
    :param path: path to the directory containing the saved_model.pb file
    """
    model_letters = keras.models.load_model(path)
    print(model_letters.summary())
    
def load_words_model(model_path, face_classifier_path):
    """
    Function to load word detection model from path.
    :param path: path to the directory containing the saved_model.pb file
    """
    model_words = keras.models.load_model(path)
    print(model_letters.summary())
    face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

def predict_letters(image, show_conf=True, show_landmarks=True, threshold=0.99):
    """
    Function that will take an image and predict if there's a letter in the image
    :param show_conf: whether to display the confidence of detection on the image
    :param show_landmarks: whether to display the keypoints of the hand on the image
    :param threshold: The minimum confidence for a detection to be returned
    :return: (letter, confidence, image)
    """
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not results.multi_hand_landmarks:
        return None, None, image
    for hand_landmarks in results.multi_hand_landmarks:
        row = []
        if show_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for i in hand_landmarks.landmark:
            row += [i.x, i.y]
        row = np.array(row).reshape(1, kp_num * 2)
        prediction = model_letters.predict(row)
        guess = np.argmax(prediction)
        if prediction[0, guess] > threshold:
            if show_conf:
                cv2.putText(image,letters[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
            return letters[guess], prediction[0, guess], image
    return None, None, image

def predict_words(image, show_conf=True, show_landmarks=True, threshold=0.98):
    """
    Function that will take an image and predict if there's a word in the image
    :param show_conf: whether to display the confidence of detection on the image
    :param show_landmarks: whether to display the keypoints of the hand on the image
    :param threshold: The minimum confidence for a detection to be returned
    :return: (word, confidence, image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    guess = None
    if results.multi_hand_landmarks:
        row = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.landmark:
                row += [i.x, i.y]
            if show_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if len(row) < kp_num * 4:
            row += [0] * (kp_num * 4 - len(row))
        if len(faces) > 0:
            row += [(faces[0, 0] + faces[0, 2]//2)/image.shape[1], (faces[0, 1] + faces[0, 3]//2)/image.shape[0]]
        else:
            row += [0.5, 0.5]
        try:
            row = np.array(row).reshape(1,kp_num * 4 + 2)
        except:
            print("Something is wrong. Skipping frame...")
            return image, ""
        prediction = model_words.predict(row)
        guess = np.argmax(prediction)
        if prediction[0, guess] > threshold:
            if show_conf:
                cv2.putText(image,words[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
            return words[guess], prediction[0, guess], image
    return None, None, image