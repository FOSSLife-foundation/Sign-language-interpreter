import keras
import cv2
import tensorflow as tf
import os
import numpy as np
from utils import *

'''
tensorflow==1.14.0
numpy==1.20.1
keras==2.3.1
h5py==2.10.0

'''

def addText(img,text,thickness=0.7,color=(255,0,0),location=(50,50)):
    cv2.putText(img,text,location,cv2.FONT_HERSHEY_SIMPLEX,thickness,color,1,cv2.LINE_AA)

def showscore(img,scores,legend,model_flag):
    treshold=0.5
    if model_flag:
        treshold = 0.7
    else:
        treshold = 0.65
    name="unknown"
    gesture_ind = np.argmax(scores)
    confidence = scores[0, gesture_ind]
    if confidence>=treshold:
        name=legend[gesture_ind]
    addText(img,name)
    print(confidence,name)
    
    return name

def pred_word(imga,input_details,output_details):
    word_interpreter.set_tensor(input_details[0]['index'], imga)
    word_interpreter.invoke()
    output_data = word_interpreter.get_tensor(output_details[0]['index'])

    return showscore(img,output_data,word_legend,model_flag=True)

def pred_letter(imga,input_details,output_details):   
    letter_interpreter.set_tensor(input_details[0]['index'], imga)
    letter_interpreter.invoke()
    output_data = letter_interpreter.get_tensor(output_details[0]['index'])

    return showscore(img,output_data,letter_legend,model_flag=False)
#Model DIRs
letter_model_dir = "./training_and_testing/edge_letter_classification/models/tflite_with_switch/letter_sobel_III.tflite"
word_model_dir = "./training_and_testing/edge_word_classification/models/egde_word_tflite_with_switch/word_sobel_II.tflite"

letter_legend={
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
        26:"switch"
    }
word_legend = {
        0:"brother",
        1:"hello",
        2:"i",
        3:"learn",
        4:"learner_1",
        5:"learner_2",
        6:"name",
        7:"no",
        8:"switch"
}

cap =cv2.VideoCapture(0)
ret , img = cap.read()
#MODEL DIR
ret,img = cap.read()
#scale = 2
#delta = 0
#ddepth = cv2.CV_16S

letter_interpreter = tf.lite.Interpreter(model_path=letter_model_dir)
letter_interpreter.allocate_tensors()
letter_input_details = letter_interpreter.get_input_details()
letter_output_details = letter_interpreter.get_output_details()

word_interpreter = tf.lite.Interpreter(model_path=word_model_dir)
word_interpreter.allocate_tensors()
word_input_details = word_interpreter.get_input_details()
word_output_details = word_interpreter.get_output_details()

word_pred = True
pred_lst = []


prev_pred=""
prev_confirmed_pred=""
k=0
while(cap.isOpened()):

    ret , img = cap.read()
    if ret==False:
        break
    
    edges = preprocess_(img)
    cv2.imshow("edges",edges)
    imga = np.expand_dims(edges, 2)
    imga = np.expand_dims(imga, 0).astype('float32')/255.
    
    if word_pred==True:
        label = pred_word(imga,word_input_details,word_output_details)

        
    else:
        label = pred_letter(imga,letter_input_details,letter_output_details)
        
        
    if prev_pred == "":
        prev_pred=label
        prev_confirmed_pred="unknown"
    elif prev_pred==label:
        k+=1
        if k>=8:
            
            k=0
            if label=="leaner_2" and prev_confirmed_pred == "leaner_1":
                addText(img,"LEARNER",1,(0,0,255),(100,100))
                
            elif label=="switch" and prev_confirmed_pred != "switch":
                
                word_pred= not word_pred
                print("Switching models")
                addText(img,label,1,(0,0,255),(100,100))   
            else:
                addText(img,label,1,(0,0,255),(100,100))    
                
            prev_confirmed_pred=label
            tts_(label)
    
    else:
        prev_pred=label
        k=0

    cv2.imshow("test",img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

