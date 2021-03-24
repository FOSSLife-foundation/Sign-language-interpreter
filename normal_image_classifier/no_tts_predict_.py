import keras
import cv2
import tensorflow as tf
import os
import numpy as np
import tts

''' 
tensorflow==1.14.0
numpy==1.20.1
keras==2.3.1
h5py==2.10.0

'''


def showscore(img,scores,legend):
    name="unknown"
    gesture_ind = np.argmax(scores)
    confidence = scores[0, gesture_ind]
    if confidence>=0.9:
        #i=scores.index(gesture)
        name=legend[gesture_ind]
        cv2.putText(img,  name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
        print(confidence,name)
    else:
        cv2.putText(img,  "unknown", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
    return name

def pred_word(imga,input_details,output_details):
    word_interpreter.set_tensor(input_details[0]['index'], imga)
    word_interpreter.invoke()
    output_data = word_interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)
    #scores = output_data.tolist()
    return showscore(img,output_data,word_legend)

def pred_letter(imga,input_details,output_details):   
    letter_interpreter.set_tensor(input_details[0]['index'], imga)
    letter_interpreter.invoke()
    output_data = letter_interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)
    #scores = output_data.tolist()
    return showscore(img,output_data,letter_legend)

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
scale = 2
delta = 0
ddepth = cv2.CV_16S

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

half=0
frames=5
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
    
    if word_pred==True:
        label = pred_word(imga,word_input_details,word_output_details)
        if label == "learner_1":
            label="unknown"
            half=1
        elif half==1:
            frames-=1
            if frames<=0:
                half=0
                pass
            elif label=="learner_2":
                label="learner"
        elif label == "learner_2":
            label="unknown"
        
    else:
        label = pred_letter(imga,letter_input_details,letter_output_details)
        
    if label!="unknown":
        
        if label=="switch":
            word_pred= not word_pred
            print("Switching models")
            if len(pred_lst)>0:
                tts.tts_(["Switching models"])
                pred_lst=[]
        else:
            pred_lst.append(label)
    if len(pred_lst)>5:
        
        if len(pred_lst)==pred_lst.count(pred_lst[0]):
            tts.tts_([pred_lst[0]])
            
        pred_lst=[]
        
        
    cv2.imshow("test",img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
