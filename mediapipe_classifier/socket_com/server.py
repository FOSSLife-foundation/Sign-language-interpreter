import socket, cv2, pickle,struct,imutils
from imutils.video import VideoStream
from funcs import tcp_read, tcp_write

# Socket Create
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = '192.168.1.8'
# host_ip = socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:",socket_address)

# Socket Accept
while True:
    client_socket,addr = server_socket.accept()
    print('GOT CONNECTION FROM:',addr)
    if client_socket:
        vid = VideoStream(src=0, usePiCamera=True).start()
        
        while True:
            img = vid.read()
            try:
                frame = imutils.resize(img,width=320)
            except:
                continue
            a = pickle.dumps(frame)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)
            cv2.imshow('TRANSMITTING VIDEO',frame)
            guess = tcp_read(client_socket)
            print(guess)
            key = cv2.waitKey(1) & 0xFF
            if key ==ord('q'):
                client_socket.close()