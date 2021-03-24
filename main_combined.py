import socket, cv2, pickle, struct, imutils, time
from imutils.video import VideoStream
from funcs import tcp_read, tcp_write
from utils import *
import time
import tensorflow as tf
import numpy as np

# Set up code for GPIO
import RPi.GPIO as GPIO
pause_btn = 27
switch_model_btn = 5
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(pause_btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(switch_model_btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)

started = False
normal_model = GPIO.input(switch_model_btn)

#*************** For Normal Image Classifier *****************

def addText(img,text,thickness=0.7,color=(255,0,0),location=(50,50)):
    cv2.putText(img,text,location,cv2.FONT_HERSHEY_SIMPLEX,thickness,color,1,cv2.LINE_AA)

def showscore(img,scores,legend,model_flag):
    treshold=0.5
    if model_flag:
        treshold = 0.7
    else:
        treshold = 0.65
    name="unknown"
    gesture_ind = np.argmax(scores)
    confidence = scores[0, gesture_ind]
    if confidence>=treshold:
        name=legend[gesture_ind]
    addText(img,name)
    print(confidence,name)
    
    return name

def pred_word(imga,input_details,output_details):
    word_interpreter.set_tensor(input_details[0]['index'], imga)
    word_interpreter.invoke()
    output_data = word_interpreter.get_tensor(output_details[0]['index'])

    return showscore(img,output_data,word_legend,model_flag=True)

def pred_letter(imga,input_details,output_details):   
    letter_interpreter.set_tensor(input_details[0]['index'], imga)
    letter_interpreter.invoke()
    output_data = letter_interpreter.get_tensor(output_details[0]['index'])

    return showscore(img,output_data,letter_legend,model_flag=False)

letter_model_dir = "/home/pi/python_client/normal_models/letter_sobel_III.tflite"
word_model_dir = "/home/pi/python_client/normal_models/word_sobel_II.tflite"

letter_legend = {
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
        26:"switch"
}
word_legend = {
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

scale = 2
delta = 0
ddepth = cv2.CV_16S

letter_interpreter = tf.lite.Interpreter(model_path=letter_model_dir)
letter_interpreter.allocate_tensors()
letter_input_details = letter_interpreter.get_input_details()
letter_output_details = letter_interpreter.get_output_details()

word_interpreter = tf.lite.Interpreter(model_path=word_model_dir)
word_interpreter.allocate_tensors()
word_input_details = word_interpreter.get_input_details()
word_output_details = word_interpreter.get_output_details()

word_pred = True
#*************************************************************
prev_pred=""
prev_confirmed_pred=""
k=0

def switch(ev = None):
    global started
    started = not started
    print("started:", started)

def switch_model(ev = None):
    global normal_model, server_socket, prev_pred, prev_confirmed_pred, k
    if not normal_model:
        try:
            server_socket.close()
        except:
            print("Socket not closeable")
    normal_model = not normal_model
    
    # Reset params
    prev_pred = ""
    prev_confirmed_pred = ""
    k = 0
    print("normal_model:", normal_model)

GPIO.add_event_detect(pause_btn, GPIO.BOTH, callback=switch, bouncetime=300)
GPIO.add_event_detect(switch_model_btn, GPIO.BOTH, callback=switch_model, bouncetime=300)

server_socket = 0
vid = VideoStream(src=0, usePiCamera=True).start()



tts_("loaded")

while True:
    if normal_model:
        if not started:
            continue
        img = vid.read()
        try:
            img = cv2.resize(img,(240,240))
        except:
            continue
        edges = preprocess_(img)
        cv2.imshow("edges",edges)
        imga = np.expand_dims(edges, 2)
        imga = np.expand_dims(imga, 0).astype('float32')/255.
        
        if word_pred==True:
            label = pred_word(imga,word_input_details,word_output_details)

            
        else:
            label = pred_letter(imga,letter_input_details,letter_output_details)
            
            
        if prev_pred == "":
            prev_pred=label
            prev_confirmed_pred="unknown"
        elif prev_pred==label:
            k+=1
            if k>=8:
                
                k=0
                if label=="leaner_2" and prev_confirmed_pred == "leaner_1":
                    addText(img,"LEARNER",1,(0,0,255),(100,100))
                    
                elif label=="switch" and prev_confirmed_pred != "switch":
                    
                    word_pred= not word_pred
                    print("Switching models")
                    addText(img,label,1,(0,0,255),(100,100))   
                else:
                    addText(img,label,1,(0,0,255),(100,100))    
                    
                prev_confirmed_pred=label
                tts_(label)
        
        else:
            prev_pred=label
            k=0

        cv2.imshow("test",img)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    else:
        cv2.destroyAllWindows()
        
        # Socket Create
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        
        host_name  = socket.gethostname()
        host_ip = '192.168.1.6'
        # host_ip = socket.gethostbyname(host_name)
        print('HOST IP:',host_ip)
        port = 9999
        socket_address = (host_ip,port)

        # Socket Accept
        while True:
            if normal_model:
                server_socket.close()
                break
            try:
                server_socket.connect(socket_address)
            except:
                print("error")
                break
            print("Connected")
            if server_socket:
                if normal_model:
                    server_socket.close()
                    break
                while True:
                    if normal_model:
                        server_socket.close()
                        break
                    if not started:
                        continue
                    img = vid.read()
                    try:
                        frame = img[:]
                    except:
                        continue
                    a = pickle.dumps(frame)
                    message = struct.pack("Q",len(a))+a
                    server_socket.sendall(message)
                    # cv2.imshow('TRANSMITTING VIDEO',frame)
                    guess = tcp_read(server_socket)
                    if len(guess) > 1:
                        if guess == prev_pred:
                            k += 1
                            if k >= 3:
                                tts_(guess)
                                k = 0
                                prev_pred = ''
                                time.sleep(1)
                        else:
                            prev_pred = guess
                            k = 0
                    else:
                        prev_pred = ''
                        k = 0
                        
                    """key = cv2.waitKey(1) & 0xFF
                    if key ==ord('q'):
                        client_socket.close()"""
                server_socket.close()
                print("closed connection")
print("exited")
cv2.destroyAllWindows()