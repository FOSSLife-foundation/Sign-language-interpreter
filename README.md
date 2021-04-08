# Sign Language Interpreter

## Introduction
The goal of this project is to build a system that acts as a translator for Sign Language, specifically American Sign Language (ASL). 

The classification can be done using two types of models
- <strong>Mediapipe-based</strong>: Here, Mediapipe is used to detect and obtain the coordinates of different landmarks of the hand. These landmarks are fed into a classification model and a prediction of the respective sign is obtained.
- <strong>Edge higlighted based</strong>: Here, the edges of shapes in the input image are highlighted and the resulting image is fed into a classification model for the prediction to be made.

Of the two methods, the Mediapipe-based model seemed to consistently outperform the Edge highlighted based model in accuracy while the opposite was true when it came to speed. Thus, the Mediapipe-based model is recommended for most cases except for when performance is a significant factor in which case the Edge highlighted model may be suitable.

## How it Works
Frames from a video feed taken from a camera would be given as input to a pair of classification models. The camera would be positioned in front of the signer and as he/she is signing, one of the models would attempt to detect letters of the alphabet while the other would attempt to detect words/expressions in ASL. Both the models would be running on a Raspberry Pi and the video feed would be taken from a Pi camera. The exact architecture in which the two models are used is to be decided based on their individual performance and their composite performance.

## Install
```console 
git clone https://github.com/FOSSLife-foundation/Sign-language-interpreter.git
cd Sign-Language-Interpreter
pip install .
```

*Note that it is necessary to locate the models and load them before using.

Models are located in the ASLInterpreter/models directory in the cloned repository.
(cloned_dir)/Sign-language-interpreter/ASLInterpreter/models/model_letters for mediapipe letter model
(cloned_dir)/Sign-language-interpreter/ASLInterpreter/models/model_words for mediapipe word model
(cloned_dir)/Sign-language-interpreter/ASLInterpreter/models/edge_letter.tflite for mediapipe letter model
(cloned_dir)/Sign-language-interpreter/ASLInterpreter/models/edge_word.tflite for mediapipe letter model

## Import structure
- ASLInterpreter
  - predict
    - edge_letter - Used to classify letters given edge highlighted frames 
      - showscore()
        - Input - model output
        - Output - predicted label
        - Used to visualize the output. This function is not required to be called by the user  
      - predict()
        - Input - Edge higlighted image of shape (240,240,1)
        - Output - model output
        - Used predict the letter given a single frame/ image    
      - load_model()
        - Input - absolute path to tflite model as a String "F:/Workspace/models/letter.tflite"
        - Output - None
        - Used to load the model    
    - edge_word - Used to classify words given edge highlighted frames
      - showscore()
      - predict()
      - load_model()
        *Almost identicle to edge_letter   
    - predict_mp - Used to classify words and letters using mediapipe
      - load_letters_model()
        - Input - Absolute path to the model folder (containing the saved_model.pb file)
        - Output - None
      - load_words_model()
        - Same as load_letters_model()
      - predict_letters()
        - Input - An image (`image`), a boolean on whether to display confidence (`show_conf`), a boolean on whether to display hand keypoints (`show_landmarks`) and a confidence threshold for detections (`threshold`)
        - Output - Classified letter, confidence of detection and the edited image
      - predict_words()
        - Same as predict_letters()
  - preprocess
    - edge_highlight
      - preprocess()
        - Input - RGB image (,,3)
        - Output - Resized edge highlighted image of shape (240,240,1)
        - Used to pre-process the video frames / images before prediction
    - mp_data_augmentation - The following methods can be used to perform data augmentation on an array of keypoints. This is useful for training one's own model.
      - flip_keypoints() - Flip keypoints along the center
      - scale_keypoints() - Scale keypoints by a random amount
      - move_keypoints() - Move keypoints by a random amount
      - rotate_keypoints() - Rotate keypoints by a random amount

## Example
Mediapipe-based classifier example
```python
from ASLInterpreter.predict.predict_mp import *
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

load_letters_model("C:/Sign-language-interpreter/ASLInterpreter/models/model_letters")

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
```
Edge highlighted based classifier
```python
from  ASLInterpreter.preprocess.edge_highlight import preprocess
#from  ASLInterpreter.predict.edge_letter import load_model,predict #this is for letter prediction
from  ASLInterpreter.predict.edge_word import load_model,predict # this is for word prediction
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

load_model("F:/WorkSpace/FOSSLIFE/Sign-Language-Interpreter/Sign-language-interpreter/ASLInterpreter/models/edge_word.tflite")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
       
    # Get edge highlited image
    edges = preprocess(image)
    
    # Get prediction
    name = predict(edges)
    
    cv2.imshow("",edges)
   
    if cv2.waitKey(30) & 0xff ==27:
        break
cv2.destroyAllWindows()
cap.release()
```

