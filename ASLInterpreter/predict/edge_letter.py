import keras
import cv2
import tensorflow as tf
import os
import numpy as np


interpreter=None
input_details=None
output_details=None

def load_model(path):
    '''
    Function to load the letter classification model
    :param img: model path
    '''
    global interpreter ,input_details ,output_details
    interpreter = tf.lite.Interpreter(model_path=path) 
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("edge-letter model loaded")

def showscore(img,scores):
    '''
    Function that can help visualize the output prediction
    :param img: image that will be annoted
    :param scores: output tensor from model
    :return: label of the prediction
    '''
    treshold = 0.65
    name="unknown"
    gesture_ind = np.argmax(scores)
    confidence = scores[0, gesture_ind]
    if confidence>=treshold:
        name=legend[gesture_ind]
    cv2.putText(img,name,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1,cv2.LINE_AA)
    print(confidence,name)
    
    return name

def predict(edges,visualize=True):
    '''
    Function that will predict the gesture
    :param edges: edge highlighted image that will be the input to the model, input shape = (240,240,1)
    :return: output tensor from the model (26,)
    '''
    imga = np.expand_dims(edges, 2)
    imga = np.expand_dims(imga, 0).astype('float32')/255.
    interpreter.set_tensor(input_details[0]['index'], imga)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if visualize:
        return showscore(edges,output_data)
    return output_data

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
        26:"Switch"
    }

