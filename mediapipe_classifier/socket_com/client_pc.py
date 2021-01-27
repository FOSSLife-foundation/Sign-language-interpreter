# lets make the client code
import socket,cv2, pickle,struct
from funcs import tcp_read, tcp_write


##################### FOR PREDICTIONS #########################
import numpy as np
import keras
model = keras.models.load_model("../models/model_4")

import imutils
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.5)

# number of keypoints per hand
kp_num = 21

count = 0
t1 = time.time()
letters = 'abcdefghijklmnopqrstuvwxy'

def process_image(image):
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
            prediction = model.predict(row)
            guess = np.argmax(prediction)
            cv2.putText(image,letters[guess] + ", confidence: " + str(round(prediction[0, guess] * 100, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if not guess:
        return image, ""
    return image, letters[guess]
###############################################################

# create socket
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.8' # paste your server ip address here
port = 9999
client_socket.connect((host_ip,port)) # a tuple
data = b""
payload_size = struct.calcsize("Q")
while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024) # 4K
        if not packet: break
        data+=packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q",packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    data  = data[msg_size:]
    frame = pickle.loads(frame_data)
    # frame = process_image(frame)
    image, guess = process_image(frame)
    cv2.imshow("RECEIVING VIDEO",image)
    tcp_write(client_socket, guess)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
client_socket.close()

hands.close()
cv2.destroyAllWindows()