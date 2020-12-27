import os
import cv2
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D 
from keras.layers import Dense, Dropout, Activation, Flatten ,BatchNormalization,GlobalMaxPool2D 
from keras.layers.experimental.preprocessing import Rescaling
from keras.optimizers import Adam
from keras import backend as K
import keras
import tensorflow as tf

def getmodel():
    '''
    builds, constructs model
    parameter

    return 
        model
    '''
    model=Sequential()

    model.add(Conv2D(64, (3,3), input_shape = (240, 240, 1), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model


def preprocess(img):
    '''
    pre-processing function passed as a parameter to ImageDaraGenerator
    parameters
        img: grayscaled image dims [x,y]
    returns
        out: numpy array of edge highligthed image of dims [x,y,1]

    '''
    scale = 2
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    out=np.expand_dims(edges,2).astype('float32')
    return out

def train(traind,vald,saved,epochs=30):
    trclasses, teclasses= [] , []
    try:
        trclasses , teclasses  =  os.listdir(traind), os.listdir(vald)
    except:
        print("Error in given dir")
        return

    aug = ImageDataGenerator(
        width_shift_range=(0.3),
        height_shift_range=(0.3),
        zoom_range=(0.2),
        rotation_range=5,
        horizontal_flip=False,
        brightness_range=[0.8,1.3],
        fill_mode="nearest",
        preprocessing_function=preprocess,
        rescale=1./255
    )
    test_datagen = ImageDataGenerator(width_shift_range=(1,1.3),preprocessing_function=preprocess,rescale=1./255)

    val_data=test_datagen.flow_from_directory(
        vald,
        class_mode="categorical",
        classes=teclasses,
        color_mode='grayscale',
        batch_size=32,
        target_size=(240, 240),
        interpolation="bilinear",
        shuffle=True, 
    )


    train_data=aug.flow_from_directory(
        traind,
        class_mode='categorical',
        classes=teclasses,
        color_mode='grayscale',
        batch_size=32,
        target_size=(240, 240),
        interpolation="bilinear"   , 
        shuffle=True
    )
    model = getmodel()
    model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
    )
    model.save(saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traindir",help="Train data directory from root",type=str)
    parser.add_argument("valdir",help="Validation data directory from root",type=str)
    parser.add_argument("savedir",help="destination of saved model",type=str)

    args = parser.parse_args()
    train(args.traindir,args.valdir,args.savedir)


if __name__ == "__main__":
    main()

