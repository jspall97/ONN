import socket
import threading
import time
import numpy as np

HOST = '127.0.0.1'  # The server's hostname or IP address


def send_data(data, PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            conn.send(str(data).encode())


def receive_data(PORT):
    data = None
    while data is None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                data = int(s.recv(3).decode())
        except (ConnectionRefusedError, ConnectionResetError):
            pass
    return data


# for _ in range(5):
#
#     send_data(1, 65432)
#
#     print('next batch started')
#
#     cam = receive_data(65433)
#
#     print('batch finished')
#     time.sleep(1)


# send_data(1)
# print('waiting')
# mess = receive_data()
# print(mess)

# for i in range(5):
#
#     mess = receive_data()
#     print(mess)
#     time.sleep(1)
#     send_data(mess+1)
#     time.sleep(1)

# n = 121
# m = 52
#
#
def update_slm(arr):
    np.save('./tools/weights1_nm.npy', arr)
    send_data(1, 65436)  # block until slm has received array
    receive_data(65435)  # block until slm has finished updating


img = np.full((n, m), 0.)
update_slm(img)

t0 = time.time()

for i in range(100):
    img += 0.01
    update_slm(img)
    print(time.time()-t0)
