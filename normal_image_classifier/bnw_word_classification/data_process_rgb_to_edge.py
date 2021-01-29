import cv2
import glob
import os
from pathlib import Path
import numpy as np

def preprocess(img):
    scale = 2
    delta = 0
    ddepth = cv2.CV_16S
    img= cv2.cvtColor(cv2.resize(img,(240,240)),cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edges

os.chdir("F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data")

paths = Path(os.curdir).glob("train/**/*.jpg")
new_path = "F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data/processed/dev"
for path in paths:
    path=str(path)
    print(path)
    img = cv2.imread(path)
    img = preprocess(img)
    cv2.imwrite(path,img)