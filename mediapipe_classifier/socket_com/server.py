import socket
import pickle
import struct
import time

###################### FOR PREDICTIONS ########################

import cv2
import numpy as np
import keras
model_letters = keras.models.load_model("../models/model_6")
print(model_letters.summary())
model_words = keras.models.load_model("../models_words/model_1")
print(model_words.summary())

import imutils
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.5)

# number of keypoints per hand
kp_num = 21

letter_threshold = 0.99
word_threshold = 0.98

###############################################################

################### FOR LETTER PREDICTIONS ####################

letters = ['_switch'] + list('abcdefghijklmnopqrstuvwxy')

def process_image_letters(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    guess = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row = []
            for i in hand_landmarks.landmark:
                row += [i.x, i.y]
            row = np.array(row).reshape(1, kp_num * 2)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            prediction = model_letters.predict(row)
            guess = np.argmax(prediction)
            if prediction[0, guess] > letter_threshold:
                cv2.putText(image,letters[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
            else:
                return image, ""
    if guess == None:
        return image, ""
    return image, letters[guess]
    
###############################################################

################## FOR WORD PREDICTIONS #######################

words = ['_switch', 'name', 'i', 'hello', 'learn', 'yes', 'no', 'thank', 'you', 'who', 'what', 'when', 'where', 'why', 'how', 'which']

face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')

def process_image_words(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    guess = None
    if results.multi_hand_landmarks:
        row = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.landmark:
                row += [i.x, i.y]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if len(row) < kp_num * 4:
            row += [0] * (kp_num * 4 - len(row))
        if len(faces) > 0:
            row += [(faces[0, 0] + faces[0, 2]//2)/1280, (faces[0, 1] + faces[0, 3]//2)/720]
        else:
            row += [0.5, 0.5]
        try:
            row = np.array(row).reshape(1,kp_num * 4 + 2)
        except:
            print("Something is wrong. Skipping frame...")
            return image, ""
        prediction = model_words.predict(row)
        guess = np.argmax(prediction)
        if prediction[0, guess] > word_threshold:
            cv2.putText(image,words[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
        else:
            return image, ""
    
    if guess == None:
        return image, ""
    return image, words[guess]
    
###############################################################


################## FOR SOCKET CONNECTION ######################
def tcp_write(s, D):
   s.send((D + '\r').encode('utf-8'))
   return

def tcp_read(s):
    a = ' '
    b = ''
    while a != '\r':
        a = s.recv(1).decode('utf-8')
        b = b + a
    return b


# Socket Create
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = '192.168.1.6'
# host_ip = socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:",socket_address)

data = b""
payload_size = struct.calcsize("Q")

###############################################################


letter_det = True

# For FPS calculation
start_time = time.time()
t = 1 # displays FPS every t second
counter = 0

while True:
    client_socket,addr = server_socket.accept()
    print("Got Connection from", addr)
    while True:
        
        while len(data) < payload_size:
            try:
                packet = client_socket.recv(4*1024)
            except:
                print("Connection closed.")
                break
            if not packet: break
            data+=packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        try:
            msg_size = struct.unpack("Q",packed_msg_size)[0]
        except:
            print("Connection closed.")
            break
        
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        frame = pickle.loads(frame_data)
        if letter_det:
            image, guess = process_image_letters(frame)
        else:
            image, guess = process_image_words(frame)
        if guess == '_switch':
            letter_det = not letter_det
        cv2.imshow("RECEIVING VIDEO",image)
        tcp_write(client_socket, guess)
        """if guess != '':
            time.sleep(1)"""
        key = cv2.waitKey(1) & 0xFF
        if key  == ord('q'):
            break    
        
        # FPS calculation
        counter += 1
        if (time.time() - start_time) > t :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
        
    client_socket.close()
    cv2.destroyAllWindows()
    print("Wait for new connection.")

hands.close()
cv2.destroyAllWindows()