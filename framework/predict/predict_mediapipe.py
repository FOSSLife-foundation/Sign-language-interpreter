import mediapipe
import cv2
import numpy as np
import pathlib

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.5)

kp_num = 21
file_dir = os.path.dirname(os.path.abspath(__file__))
model_letters = keras.models.load_model("../models/model_6")
model_words = keras.models.load_model("../models_words/model_1")

def set_letters_model_dir(path)
    model_letters = keras.models.load_model(path)
    
def set_words_model_dir(path)
    model_words = keras.models.load_model(path)

def predict_letters_from_image(image, show_conf=True, show_landmarks=True, threshold=0.99):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not results.multi_hand_landmarks:
        return False
    for hand_landmarks in results.multi_hand_landmarks:
        row = []
        if show_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for i in hand_landmarks.landmark:
            row += [i.x, i.y]
        row = np.array(row).reshape(1, kp_num * 2)
        prediction = model.predict(row)
        guess = np.argmax(prediction)
        if prediction[0, guess] > threshold:
            if show_conf:
                cv2.putText(image,letters[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
            return letters[guess], prediction[0, guess], image
    return None, None, image

def predict_words_from_image(image, show_conf=True, show_landmarks=True, threshold=0.98):
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
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if len(row) < kp_num * 4:
            row += [0] * (kp_num * 4 - len(row))
        if len(faces) > 0:
            row += [(faces[0, 0] + faces[0, 2]//2)/1280, (faces[0, 1] + faces[0, 3]//2)/720]
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
            cv2.putText(image,words[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
        else:
            return image, ""
    
    if guess == None:
        return image, ""
    return image, words[guess]