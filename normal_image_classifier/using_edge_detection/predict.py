import keras
import cv2
import tensorflow as tf
import os
import numpy as np
#import time


def showscore(img,scores):
    
    gesture = max(scores)
    if gesture>=0.45:
        i=scores.index(gesture)
        name=legend[i]
        cv2.putText(img,  name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        print(gesture,name)
    else:
        cv2.putText(img,  "unknown", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

legend = {
        0:"A",
        1:"B",
        2:"C",
        3:"D",
        4:"E",
        5:"F",
        6:"G",
        7:"H",
        8:"I",
        9:"J",
        10:"K",
        11:"L",
        12:"M",
        13:"N",
        14:"O",
        15:"P",
        16:"Q",
        17:"R",
        18:"S",
        19:"T",
        20:"U",
        21:"V",
        22:"W",
        23:"X",
        24:"Y",
        25:"Z",
        26:"del",
        27:"space",
        28:"none"
    }

cap =cv2.VideoCapture(0)
ret , img = cap.read()
model_dir="F:/WorkSpace/Sign_Language_Testing/models/model-19-sobel"
ret,img = cap.read()
scale = 2
delta = 0
ddepth = cv2.CV_16S
model_loaded = keras.models.load_model(model_dir)

while(True):
    #st = time.time()
    ret , img = cap.read()
    if ret==False:
        break
    gray= cv2.cvtColor(cv2.resize(img,(240,240)),cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("edges",edges)
    imga = np.expand_dims(edges, 2)
    imga = np.expand_dims(imga, 0).astype('float32')/255.
    predictions = model_loaded.predict(imga)
    scores = predictions.tolist()
    showscore(img,scores[0])
    cv2.imshow("test",img)
    #end = time.time()
    #print(str(end-st))
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()