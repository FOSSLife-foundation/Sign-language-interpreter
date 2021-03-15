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

#Change directory
os.chdir("F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data")
#folder structure to raw rgb images: rgb/<label>/<imagename>.jpg
paths = Path(os.curdir).glob("rgb/**/*.jpg")
new_path = "F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data/edge"
for path in paths:
    path=str(path)
    print(path)
    lst_path = path.split("\\")
    dest_path  = new_path+"/"+lst_path[-2]
    save_path = new_path+"/"+lst_path[-2]+"/"+lst_path[-1]
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    img = cv2.imread(path)
    img = preprocess(img)
    cv2.imwrite(save_path,img)