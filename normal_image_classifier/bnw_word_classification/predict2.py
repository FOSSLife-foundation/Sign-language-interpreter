import cv2
import keras
import numpy as np
from scipy.ndimage import rotate
from preprocess import to_edge
from preprocess import to_binary


def showscore(img,scores):
    gesture = max(scores)
    if gesture>=0.6:
        i=scores.index(gesture)
        name=legend[i]
        cv2.putText(img,  name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        print(gesture,name)
    else:
        cv2.putText(img,  "unknown", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)


#cap = cv2.VideoCapture("F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/testing/test2.mp4")
cap=cv2.VideoCapture(0)

#model = keras.models.load_model(model_dir)
model = keras.models.load_model('F:/WorkSpace/Sign_Language_Testing/Words_testing/st_end/models/edge/sobel_bnw_3_II')

legend = {
        0:"Brother",
        1:"Hello",
        2:"i",
        3:"learn",
        4:"learner_1",
        5:"learner_2",
        6:"name",
        7:"no"
}
scale = 2
delta = 0
ddepth = cv2.CV_16S


while (cap.isOpened()):
    ret , img = cap.read()
    
    img = cv2.cvtColor(cv2.resize(img,(240,240)),cv2.COLOR_BGR2GRAY)
    img = rotate(img, 90)
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    (thresh, bnw) = cv2.threshold(edges, 120, 255, cv2.THRESH_BINARY)
    print(bnw.shape)
    x = np.expand_dims(bnw, axis=2)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')/255.
    preds = model.predict(x)
    scores = preds.tolist()
    showscore(img,scores[0])
    cv2.imshow("gray",img)
    cv2.imshow("binary",bnw)
    print(scores)

    key = cv2.waitKey(3)
    if key & 0xff == 27:
        break
