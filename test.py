from ASLInterpreter.predict.predict_mediapipe import *
from ASLInterpreter.predict.edge_word import *
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

load_letters_model("C:/Users/ASUS/Desktop/Automation Challenge/Sign-language-interpreter/ASLInterpreter/models/model_letters")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    letter, confidence, image = predict_letters_from_image(image)
    if letter:
        print(letter, ":", confidence)
    
    cv2.imshow('Test', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break