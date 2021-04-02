import cv2

def preprocess(img,size=(240,240),scale=2,delta=0,ddepth=cv2.CV_16S):
    '''
    :param img: input raw frame
    :return: edge highlighted image shape (240,240,1) by default
    '''
    gray= cv2.cvtColor(cv2.resize(img,size),cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edges