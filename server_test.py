import socket
import threading
import time

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)


# def receive_data():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         conn, addr = s.accept()
#         with conn:
#             data = int(conn.recv(3).decode())
#     return data

#
# def send_data(data):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         conn, addr = s.accept()
#         with conn:
#             conn.send(str(data).encode())


def send_data(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            conn.send(str(data).encode())


def receive_data():
    data = None
    while data is None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                data = int(s.recv(3).decode())
        except (ConnectionRefusedError, ConnectionResetError):
            pass
    return data


# mess = 0
#
# send_data(1)
#
# for i in range(5):
#
#     send_data(mess+1)
#     # print('sent')
#     time.sleep(1)
#     mess = receive_data()
#     print(mess)
#     time.sleep(1)


# slm_locked = 1
#
#
# for i in range(5):
#
#     while slm_locked:
#         slm_locked = receive_data()
#
#     print('unlocked, updating slm')
#     time.sleep(2)
#
#     print('slm updated, locking')
#     slm_locked = 1
#     send_data(slm_locked)
#
#     print()

