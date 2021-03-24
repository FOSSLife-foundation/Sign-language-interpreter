import socket, cv2, pickle, struct, imutils, time
from imutils.video import VideoStream
from funcs import tcp_read, tcp_write
import time

import pyttsx3
engine = pyttsx3.init()

def speak(speak_txt):
    print(speak_txt)
    engine.say(speak_txt)
    engine.runAndWait()

# Set up code for GPIO
import RPi.GPIO as GPIO
pause_btn = 27
switch_model_btn = 5
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(pause_btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(switch_model_btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)

started = True
normal_model = GPIO.input(switch_model_btn)

def switch(ev = None):
    global started
    started = not started
    print("started:", started)

def switch_model(ev = None):
    global normal_model, server_socket
    if not normal_model:
        try:
            server_socket.close()
        except:
            print("Socket not closeable")
    normal_model = not normal_model
    print("normal_model:", normal_model)

GPIO.add_event_detect(pause_btn, GPIO.BOTH, callback=switch, bouncetime=300)
GPIO.add_event_detect(switch_model_btn, GPIO.BOTH, callback=switch_model, bouncetime=300)

server_socket = 0
vid = VideoStream(src=0, usePiCamera=True).start()

while True:
    if normal_model:
        pass
    else:
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
                        speak(guess)
                        time.sleep(1)
                    """key = cv2.waitKey(1) & 0xFF
                    if key ==ord('q'):
                        client_socket.close()"""
                server_socket.close()
                print("closed connection")
print("exited")
