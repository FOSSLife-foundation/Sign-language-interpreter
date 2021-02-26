import cv2
import glob
import os
from pathlib import Path
import numpy as np

def preprocess(img):
    out  = cv2.resize(img,(240,240))
    return out

os.chdir("F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data")
#Relative folder structure to raw data : <data_folder>/<label>/<image_name>.jpg
paths = Path(os.curdir).glob("image/**/*.jpg")
#set path to resized data
new_path = "F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/data/rgb"
for path in paths:
    path=str(path)
    path_lst=path.split("\\")
    label = path_lst[-2]
    file_name = path_lst[-1]
    print(label)
    img = cv2.imread(path)
    img = preprocess(img)
    save_path = new_path+"/"+label+"/26_webcam_"+file_name
    cv2.imwrite(save_path,img)