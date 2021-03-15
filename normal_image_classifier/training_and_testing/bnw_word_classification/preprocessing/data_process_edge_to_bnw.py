import cv2
import glob
import os
from pathlib import Path
import numpy as np

def preprocess(img):
    (thresh, bnw) = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return bnw

os.chdir("F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data")

paths = Path(os.curdir).glob("train/**/*.jpg")
new_path = "F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data/bnw"
for path in paths:
    path=str(path)
    #print(path)
    lst = path.split("\\")
    img = cv2.imread(path)
    img = preprocess(img)
    dest=new_path+"/"+lst[-2]+"/"+lst[-1]
    print(dest)
    cv2.imwrite(dest,img)