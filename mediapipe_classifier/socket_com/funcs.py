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