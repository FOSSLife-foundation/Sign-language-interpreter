import keras
import cv2
import tensorflow as tf
import os
import numpy as np
''' 
tensorflow==1.14.0
numpy==1.20.1
keras==2.3.1
h5py==2.10.0

'''


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
        0:"Brother",
        1:"Hello",
        2:"i",
        3:"learn",
        4:"learner_1",
        5:"learner_2",
        6:"name",
        7:"no"
}

cap =cv2.VideoCapture(0)
ret , img = cap.read()
#MODEL DIR
model_dir="/home/pi/Desktop/Chamodya-ka/Sign-language-interpreter/normal_image_classifier/edge_word_classification/models/edge_word_tflite/edge_word.tflite"
ret,img = cap.read()
scale = 2
delta = 0
ddepth = cv2.CV_16S

interpreter = tf.lite.Interpreter(model_path=model_dir)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


while(cap.isOpened()):

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
    interpreter.set_tensor(input_details[0]['index'], imga)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    
    scores = output_data.tolist()
    showscore(img,scores[0])
    cv2.imshow("test",img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()