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

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(240, 240, 1),padding="same")) 
    model.add(MaxPooling2D((5, 5)))

    model.add(Conv2D(128, (3, 3),activation='relu'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(256, (3 ,3), activation='relu') )
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3 ,3), activation='relu') )
    model.add(MaxPooling2D((2, 2)))


    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model



def train(traind,vald,saved,epochs=30):
    trclasses, teclasses= [] , []
    try:
        trclasses , teclasses  =  os.listdir(traind), os.listdir(vald)
    except:
        print("Error in given dir")
        return

    aug = ImageDataGenerator(
        rotation_range=(15),
        horizontal_flip=False,
        zoom_range=(0.9,1.2),
        rescale=1./255
        )

    

    val_data=aug.flow_from_directory(
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
    parser.add_argument("traindir",help="Preprocessed train data directory from root",type=str)
    parser.add_argument("valdir",help="Preprocessed validation data directory from root",type=str)
    parser.add_argument("savedir",help="destination of saved model",type=str)

    args = parser.parse_args()
    train(args.traindir,args.valdir,args.savedir)


if __name__ == "__main__":
    main()
