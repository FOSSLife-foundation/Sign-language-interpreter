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
    Function to load the word classification model
    :param img: model path
    '''
    global interpreter ,input_details ,output_details
    interpreter = tf.lite.Interpreter(model_path=path) 
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("edge-word model loaded")


def showscore(img,scores):
    '''
    Function that can help visualize the output prediction
    :param img: image that will be annoted
    :param scores: output tensor from model
    :return: label of the prediction
    '''
    treshold = 0.75
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
    :return: output tensor from the model (9,)
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

