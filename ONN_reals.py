from pypylon import pylon
from pypylon import genicam
import time
import numpy as np
import sys
import threading
import time
import sys
import signal
import numpy as np
import cupy as cp
from glumpy import app
from make_dmd_image import make_dmd_rgb, make_dmd_image
from MNIST import map_vec_to_arr
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm1_image import make_slm_rgb, slm_rm_case_10
from multiprocessing import Process, Pipe
from ANN import DNN, accuracy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
from termcolor import colored
import queue
from collections import deque
import ticking
from glumpy.app import clock
from pylon import view_camera, Camera
import serial

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])
l = int(dims[2])
dmd_block_w = 15

# inputs = loadmat('./tools/MNIST digit - subsampled - 121.mat')
#
# trainX_raw = inputs['trainX']
# trainY_raw = inputs['trainY']
# testX_raw = inputs['testX']
# testY_raw = inputs['testY']
#
# num_train = 60000
# num_test = 10000
#
# trainY = np.zeros((num_train, 10))
# testY = np.zeros((num_test, 10))
#
# for i in range(num_train):
#     trainY[i, trainY_raw[0, i]] = 1
#
# for i in range(num_test):
#     testY[i, testY_raw[0, i]] = 1
#
# trainX = np.empty((num_train, 121))
# for i in range(num_train):
#     trainX_k = trainX_raw[i, :] - trainX_raw[i, :].min()
#     trainX_k = trainX_k / trainX_k.max()
#     trainX[i, :] = trainX_k
#
# testX = np.empty((num_test, 121))
# for i in range(num_test):
#     testX_k = testX_raw[i, :] - testX_raw[i, :].min()
#     testX_k = testX_k / testX_k.max()
#     testX[i, :] = testX_k
#
# random.Random(0).shuffle(trainX)
# random.Random(0).shuffle(trainY)
# random.Random(0).shuffle(testX)
# random.Random(0).shuffle(testY)
#
# valX = testX[:5000, :].copy()
# testX = testX[5000:, :].copy()
#
# valY = testY[:5000, :].copy()
# testY = testY[5000:, :].copy()


def keyboardinterrupthandler(signal, frame):
    context.pop()
    print('keyboard interupt - closed')
    exit(0)


signal.signal(signal.SIGINT, keyboardinterrupthandler)

actual_uppers_arr_1024 = np.load('./tools/actual_uppers_arr_1024_interp.npy')
uppers1_nm = actual_uppers_arr_1024[-1, ...].copy()

# actual_uppers_arr_1024_T = np.transpose(actual_uppers_arr_1024, (2, 0, 1))
gpu_actual_uppers_arr_1024 = cp.asarray(actual_uppers_arr_1024)

phase_offset = cp.load('tools/phase_offset.npy')
# phase_offset = cp.flip(phase_offset, axis=1)


def update_slm(arr, lut=False, stag=False, ref=True):

    if arr.shape[1] == m-3:
        temp = np.empty((n, m-1))
        temp[:, 1:-1] = arr.copy()
        temp = np.insert(temp, m // 2, np.zeros(n), 1)
        arr = temp

    if lut:

        gpu_arr = cp.asarray(arr)
        map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_1024 - gpu_arr), axis=0)
        arr_A = cp.linspace(-1, 1, 1024)[map_indx]
        # arr_A = cp.flip(arr_A, axis=1)
        # arr_A = cp.flip(arr_A, axis=0)

    else:
        arr_A = cp.asarray(arr)

    arr_phi = cp.asarray(np.angle(arr)) + phase_offset

    if ref:
        # arr_A[:, -1] = 1
        # arr_phi[:, -1] = np.pi / 2
        #
        # arr_A[:, 0] = 1  # ampl_norm_val
        # arr_phi[:, 0] = phase_offset[:, 0]

        arr_A[:, 0] = 1
        arr_phi[:, 0] = phase_offset[:, 0] + np.pi / 2

        arr_A[:, m//2] = 0.1  # ampl_norm_val
        arr_phi[:, m//2] = phase_offset[:, m//2]

        arr_A[:, -1] = 1
        arr_phi[:, -1] = phase_offset[:, -1]

    img = make_slm_rgb(arr_A, arr_phi, stagger=stag)
    slm.updateArray(img)
    # time.sleep(0.7)


def dmd_one_frame(arr):
    img = make_dmd_image(arr)
    frame = make_dmd_rgb([img for _ in range(24)])
    return [frame]


x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
y_center_indxs = np.load('./tools/y_center_indxs.npy')

area_height = 4
half_height = area_height // 2


def find_spot_ampls(arr):

    def spot_s(ii):
        y_center_i = y_center_indxs[ii]
        return np.s_[:, y_center_i - half_height:y_center_i + half_height + 1,
                     x_edge_indxs[2 * ii]:x_edge_indxs[2 * ii + 1]]

    spots_dict = {}

    for spot_num in range(m):
        spot = arr[spot_s(spot_num)]

        mask = spot < 3
        spot -= 2
        spot[mask] = 0

        spots_dict[spot_num] = spot

    spot_powers = np.array([spots_dict[ii].mean(axis=(1, 2)) for ii in range(m)])

    spot_ampls = np.sqrt(spot_powers)

    # spot_ampls = np.flip(spot_ampls.T, axis=1)

    return spot_ampls.T


ref_slice = np.s_[y_center_indxs[-1] - half_height: y_center_indxs[-1] + half_height + 1,
                  x_edge_indxs[-2]:x_edge_indxs[-1]]


if __name__ == '__main__':

    ################
    # SLM display #
    ################

    slm = SLMdisplay(0)

    ################
    # DMD display #
    ################

    backend = app.use('glfw')

    window = app.Window(1920, 1080, fullscreen=0, decoration=0)
    window.set_position(-1920, 0)
    window.activate()
    window.show()

    @window.event
    def on_draw(dt):
        global cp_arr, frame_count, target_frames
        window_on_draw(window, screen, cuda_buffer, cp_arr)
        frame_count += 1
        cp_arr = target_frames[frame_count % len(target_frames)]

    screen, cuda_buffer, context = setup(1920, 1080)

    zeros = np.zeros((n, m))
    zeros[:, -1] = 1
    null_frame = dmd_one_frame(zeros)[0]
    null_frames = [null_frame for _ in range(10)]

    full_frame = dmd_one_frame(np.ones((n, m)))[0]
    full_frames = [full_frame for _ in range(10)]

    ###################
    # locking trigger #
    ###################

    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM5'
    ser.timeout = 1
    ser.open()

    ser.flushInput()
    ser.write("y".encode())

    def relock():
        ser.write("y".encode())
        time.sleep(0.5)
        ser.write("y".encode())
        time.sleep(0.5)

    locked = 0

    def lock_read():
        global locked
        if ser.in_waiting > 0:
            locked = int(ser.read(ser.in_waiting).decode()[-1])

    ###################
    # set brightness #
    ###################

    ampl_norm_val = 0.1
    arr = np.full((n, m), ampl_norm_val)
    update_slm(arr, lut=False, stag=False, ref=True)

    target_frames = dmd_one_frame(np.ones((n, m)))
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(framerate=0, framecount=1)

    ser.write("y".encode())
    time.sleep(3)

    # input()

    # view_camera()

    #####################
    # find norm values #
    #####################

    # print("SIGNAL ONLY")
    # view_camera()
    #
    # cam = Camera()
    # cam.capture()
    # sig_frame = cam.arr.copy()[np.newaxis, ...]
    # A0 = find_spot_ampls(sig_frame).mean(axis=0)
    # np.save('./tools/A0.npy', A0)
    # print("signal: ", A0[1:-1].max())
    # print(A0)
    # cam.close()
    #
    # print("REF ONLY")
    # view_camera()
    #
    # cam = Camera()
    # cam.capture()
    # ref_frame = cam.arr.copy()[np.newaxis, ...]
    # Aref = find_spot_ampls(ref_frame).mean(axis=0)
    # np.save('./tools/Aref.npy', Aref)
    # print("ref: ", Aref[1:-1].max())
    # print(Aref)
    # cam.close()
    #
    # # breakpoint()
    #
    # print("RELOCK")
    # input("ready")
    # relock()
    # view_camera()


    # print("REF")
    # cam = Camera()
    #
    # arr = np.full((n, m), 0)
    # update_slm(arr, lut=False, stag=False, ref=False)
    # time.sleep(1)
    #
    # cam.capture()
    # frame = cam.arr.copy()[np.newaxis, ...]
    # Aref = find_spot_ampls(frame).mean(axis=0)
    # np.save('./tools/Aref.npy', Aref)
    # print("ref: ", Aref[1:-1].max())
    #
    # print("SIG")
    #
    # arr = uppers1_nm.copy() * ampl_norm_val
    # update_slm(arr, lut=True, stag=False, ref=True)
    # time.sleep(2)
    #
    # cam.capture()
    # frame = cam.arr.copy()[np.newaxis, ...]
    # A0 = find_spot_ampls(frame).mean(axis=0) - Aref
    # np.save('./tools/A0.npy', A0)
    # print("signal: ", A0[1:-1].max())
    # cam.close()
    #
    # print("RELOCK")
    #
    # arr = uppers1_nm.copy() * ampl_norm_val
    # update_slm(arr, lut=True, stag=False, ref=True)
    # time.sleep(1)
    #
    # # relock()
    # time.sleep(1)
    #
    # view_camera()

    # A0 = np.load('./tools/A0.npy')
    # Aref = np.load('./tools/Aref.npy')

    # breakpoint()

    # find theory output vector corresponding to SLM at ampl_norm_val
    k = int(np.argmin(np.abs(np.linspace(-1, 1, 1024) - ampl_norm_val)))
    ref_row = m//2

    z0 = actual_uppers_arr_1024[k, ...].copy().sum(axis=0)
    z0_refrow = z0[ref_row]


    print()
    print('############')
    print()

    ################
    # Pylon camera #
    ################

    imageWindow = pylon.PylonImageWindow()
    imageWindow.Create(1)

    # Create an instant camera object with the camera device found first.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Print the model name of the camera.
    print("Using device ", camera.GetDeviceInfo().GetModelName())
    print()

    pylon.FeaturePersistence.Load("./tools/pylon_settings.pfs", camera.GetNodeMap())

    class CaptureProcess(pylon.ImageEventHandler):
        def __init__(self):
            super().__init__()

            self.all_frames = []
            self.frames = []
            self.timestamps = []

        def OnImageGrabbed(self, cam, grab_result):
            if grab_result.GrabSucceeded():

                image = grab_result.GetArray()
                # timestamp = grab_result.ChunkTimestamp.Value

                self.all_frames.append(image)

                if image[ref_slice].mean() > 180:
                    self.frames.append(image)
                    # self.timestamps.append(timestamp)

                self.frames = self.frames[-800:]
                self.all_frames = self.all_frames[-800:]
                # self.timestamps = self.timestamps[-1000:]

                # imageWindow.SetImage(grab_result)
                # imageWindow.Show()

    # register the background handler and start grabbing using background pylon thread
    capture = CaptureProcess()
    camera.RegisterImageEventHandler(capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
    time.sleep(1)

    # def find_norm_vals():
    #
    #     time.sleep(1)
    #     lock_read()
    #     if not locked:
    #         ser.write("y".encode())
    #
    #     global target_frames, cp_arr, frame_count
    #
    #     target_frames = dmd_one_frame(np.ones((n, m)))
    #     cp_arr = target_frames[0]
    #     frame_count = 0
    #     app.run(framerate=0, framecount=1)
    #
    #     print("REF")
    #     capture.all_frames = []
    #     arr = np.full((n, m), 0)
    #     update_slm(arr, lut=False, stag=False, ref=False)
    #     time.sleep(1)
    #     ref_frames = np.array(capture.all_frames.copy())
    #     Aref = find_spot_ampls(ref_frames).mean(axis=0)
    #     np.save('./tools/Aref.npy', Aref)
    #     print("ref: ", Aref[1:-1].max())
    #     print(Aref)
    #
    #     print("\nSIGNAL")
    #
    #     arr = uppers1_nm.copy() * ampl_norm_val
    #     update_slm(arr, lut=True, stag=False, ref=True)
    #     time.sleep(1)
    #
    #     relock()
    #
    #     while not locked:
    #         lock_read()
    #         time.sleep(1)
    #
    #     capture.all_frames = []
    #     time.sleep(1)
    #     sig_frames = np.array(capture.all_frames.copy())
    #     Aboth = find_spot_ampls(sig_frames).mean(axis=0)
    #     A0 = Aboth - Aref
    #     np.save('./tools/A0.npy', A0)
    #     print("signal: ", A0[1:-1].max())
    #     print(Aboth)
    #     print(A0)
    #
    #     return A0, Aref

    #
    # for _ in range(5):
    #     A0, Aref = find_norm_vals()
    #     time.sleep(2)
    #
    # breakpoint()
    # view_camera()

    # Anorm = np.load('./tools/Anorm.npy')

    ###########################
    # loop batches and epochs #
    ###########################

    num_batches = 250
    num_frames = 10
    batch_size = num_frames * 24
    num_epochs = 10

    uppers1_nm_ann = uppers1_nm.copy()
    uppers1_nm_ann = np.delete(uppers1_nm_ann, [0, m-1, m // 2], 1)

    w1 = np.random.normal(0, 0.1, (n, m-3))

    # w1 = uppers1_nm_ann*0.2

    w2 = np.random.normal(0, 0.5, (m-3, l))

    m_dw1 = np.zeros((n, m-3))
    v_dw1 = np.zeros((n, m-3))
    m_dw2 = np.zeros((m-3, l))
    v_dw2 = np.zeros((m-3, l))
    beta1 = 0.9
    beta2 = 0.999
    adam_params = (m_dw1, v_dw1, m_dw2, v_dw2, beta1, beta2)

    # w1 = np.load('D:/MNIST/w1/w1_epoch_{}_batch_{}.npy'.format(9, 249))
    # w2 = np.load('D:/MNIST/w2/w2_epoch_{}_batch_{}.npy'.format(9, 249))
    # adam_params = list(np.load('D:/MNIST/adam_params.npy', allow_pickle=True))

    w1 = np.clip(w1, -uppers1_nm_ann, uppers1_nm_ann)
    update_slm(w1, lut=True, stag=False, ref=True)
    time.sleep(0.7)

    best_w1 = w1.copy()

    dnn = DNN(*adam_params, w1_0=w1, w2_0=w2, batch_size=batch_size, num_batches=num_batches, lr=5e-3)

    loss = [20]
    accs = []

    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
    axs3.set_ylim(0, 100)
    axs3.set_xlim(0, 30)
    axs3.plot(loss, linestyle='-', marker='x', c='b')

    fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
    axs4.set_ylim(0, 5)
    axs4.set_xlim(0, 1500)
    axs4.plot(accs, linestyle='-', marker='', c='r')

    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.001)

    loop_clock = clock.Clock()
    loop_clock.tick()

    all_xs = {i: np.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/xs/xs_{}.npy'.format(i)) for i in
              range(2500)}
    all_ys = {i: np.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/ys/ys_{}.npy'.format(i)) for i in
              range(2500)}

    for epoch_num in range(num_epochs):

        t0 = time.time()

        epoch_rand_indxs = np.arange(60000 // 24)
        random.Random(epoch_num).shuffle(epoch_rand_indxs)

        batch_indxs = []
        for i in range(num_batches):
            batch_indxs.append(epoch_rand_indxs[i * num_frames: (i + 1) * num_frames])

        epoch_xs = []
        epoch_ys = []

        for batch_num in range(250):
            xs_i = np.array([all_xs[i] for i in batch_indxs[batch_num]]).reshape((240, n))
            ys_i = np.array([all_ys[i] for i in batch_indxs[batch_num]]).reshape((240, 10))
            epoch_xs.append(xs_i)
            epoch_ys.append(ys_i)

        # for i in range(num_batches):
        #
        #     for j, indx in enumerate(batch_indxs[i]):
        #         if j == 0:
        #             xs_i = np.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/xs/xs_{}.npy'.format(indx))
        #             ys_i = np.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/ys/ys_{}.npy'.format(indx))
        #         else:
        #             xs_i = np.concatenate((xs_i, np.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/'
        #                                                  'xs/xs_{}.npy'.format(indx))))
        #             ys_i = np.concatenate((ys_i, np.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/'
        #                                                  'ys/ys_{}.npy'.format(indx))))
        #     epoch_xs.append(xs_i)
        #     epoch_ys.append(ys_i)

        # A0, Aref = find_norm_vals()

        # DO A FEW 'WARM UP' RUNS

        # target_frames = [cp.load('./MNIST/trainX_rgb_frames_dualref/rgb24_{}.npy'.format(j))[0, ...]
        #                  for j in batch_indxs[0]]

        target_frames = [cp.load('./MNIST/trainX_rgb_frames_dualref/rgb24_{}.npy'.format(j))[0, ...]
                         for j in batch_indxs[0]]

        for _ in range(2):
            target_frames[0:0] = [null_frame]
            target_frames.extend([null_frame])

        fc = len(target_frames) - 1
        cp_arr = target_frames[0]
        frame_count = 0

        for _ in range(2):
            capture.frames = []
            capture.all_frames = []

            app.run(framerate=0, framecount=fc)
            time.sleep(1)
            update_slm(w1, lut=True)
            time.sleep(1)
            loop_clock.tick()

        print()

        for batch_num in range(num_batches):

            # target_frames = [cp.load('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/frames/rgb24_{}.npy'
            #                          .format(j))[0, ...] for j in batch_indxs[batch_num]]

            target_frames = [cp.load('./MNIST/trainX_rgb_frames_dualref/rgb24_{}.npy'.format(j))[0, ...]
                             for j in batch_indxs[batch_num]]

            # target_frames = [cp.array(dmd_one_frame(np.ones((n, m)))[0]) for j in range(10)]


            dt = loop_clock.tick()
            print(batch_num, dt)

            for _ in range(2):
                target_frames[0:0] = [null_frame]
                target_frames.extend([null_frame])

            fc = len(target_frames)-1
            cp_arr = target_frames[0]
            frame_count = 0

            for _ in range(100):
                lock_read()
                if locked:

                    capture.frames = []
                    capture.all_frames = []
                    # capture.timestamps = []

                    app.run(framerate=0, framecount=fc)

                    time.sleep(0.1)

                    frames = np.array(capture.frames.copy())
                    all_frames = np.array(capture.all_frames.copy())
                    # times = np.array(capture.timestamps.copy())[:frames.shape[0], ...]

                    break

                else:
                    print('NOT LOCKED, RETRYING')
                    time.sleep(3)
                    # A0, Aref = find_norm_vals()
                    # time.sleep(1)

            # capture.frames = []
            # capture.all_frames = []
            # # capture.timestamps = []
            #
            # app.run(framerate=0, framecount=fc)
            # time.sleep(0.1)
            #
            # frames = np.array(capture.frames.copy())
            # all_frames = np.array(capture.all_frames.copy())
            # # times = np.array(capture.timestamps.copy())[:frames.shape[0], ...]

            capture.frames = []
            capture.all_frames = []
            # capture.timestamps = []

            np.save('D:/MNIST/pylon_captures/frames/batch_{}.npy'.format(batch_num), all_frames)
            # del target_frames

            dt = loop_clock.tick()
            print(batch_num, dt)

            Aref = find_spot_ampls(all_frames[:20, ...]).mean(axis=0)

            print(frames.shape[0], all_frames.shape[0])

            # ref_val = ampls[0, m//2]
            # noise = 0.2
            # indxs = np.where(np.abs(ampls[:, m//2] - ref_val) > noise)[0]
            # start_indx = indxs[0]
            # end_indx = indxs[0]
            # ampls = ampls[start_indx:start_indx+240, :]

            # diffs = np.diff(ampls)
            #
            # indxs = np.abs(diffs[:, m // 2]) > 0.1
            # sindx = np.where(indxs)[0][0]
            # eindx = np.where(indxs)[0][-1] + 1
            # ampls = ampls[sindx:eindx, :]


            if frames.shape[0] == 240:

                ampls = find_spot_ampls(frames)

                np.save('D:/MNIST/pylon_captures/ampls/batch_{}.npy'.format(batch_num), ampls)

                ref_row = m//2

                A0 = (ampls[:, ref_row].copy() - Aref[ref_row])[:, np.newaxis]

                z1s = (ampls - Aref)*z0_refrow/A0

                z1s = np.delete(z1s, [0, m-1, ref_row], 1)

                xs = epoch_xs[batch_num]
                ys = epoch_ys[batch_num]

                # xs = np.full((240, n), 1.)

                theories = (xs[:, None] * w1.T).transpose(0, 2, 1).sum(axis=1)

                # theories = (w1.T).transpose(0, 2, 1).sum(axis=1)

                dnn.feedforward(z1s)
                # dnn.feedforward(theories)

                dnn.backprop(xs, ys)

                w1 = dnn.w1.copy()

                update_slm(w1, lut=True)
                # time.sleep(0.7)

                # if dnn.loss < loss[-1]:
                #     best_w1 = w1.copy()

                loss.append(dnn.loss)
                print(colored('loss : {:.2f}'.format(dnn.loss), 'green'))

                np.save('D:/MNIST/loss.npy', np.array(loss))
                new_adam_params = np.array([dnn.m_dw1, dnn.v_dw1, dnn.m_dw2, dnn.v_dw2, dnn.beta1, dnn.beta2])
                np.save('D:/MNIST/adam_params.npy', new_adam_params)

            else:
                print(colored('wrong num frames: {}'.format(frames.shape[0]), 'red'))
                z1s = np.full((batch_size, m-3), np.nan)
                theories = np.full((batch_size, m-3), np.nan)

            np.save('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/measured/'
                    'measured_arr_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), z1s)
            np.save('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/theory/'
                    'theory_arr_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), theories)

            np.save('D:/MNIST/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1))
            np.save('D:/MNIST/w2/w2_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w2))

            dt = loop_clock.tick()

            # if dt > 0.6:
            #     time.sleep(2)

            print(batch_num, dt)
            print()

        # dnn.w1 = best_w1.copy()

        # update_slm(dnn.w1, lut=True)
        time.sleep(0.7)

        ####################
        # EPOCH VALIDATION #
        ####################

        val_ampls = np.full((4800, m), np.nan)
        xs = np.load('D:/MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/xs.npy')
        ys = np.load('D:/MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/ys.npy')

        for val_batch in range(20):

            frs = cp.load('D:/MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/frames/rgb24_{}.npy'.format(val_batch))
            target_frames = [frs[i, ...] for i in range(10)]

            for _ in range(2):
                target_frames[0:0] = [null_frame]
                target_frames.extend([null_frame])

            fc = len(target_frames) - 1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []
            capture.all_frames = []
            # capture.timestamps = []

            # time.sleep(0.05)

            app.run(framerate=0, framecount=fc)
            time.sleep(0.1)

            frames = np.array(capture.frames.copy())
            all_frames = np.array(capture.all_frames.copy())
            np.save('D:/MNIST/pylon_captures/validation/frames/batch_{}.npy'.format(val_batch), all_frames)

            # try:
            #     ref_val = ampls[0, m//2]
            #     noise = 0.2
            #     indxs = np.where(np.abs(ampls[:, 17] - ref_val) > noise)[0]
            #     start_indx = indxs[0]
            #     ampls = ampls[start_indx:start_indx+240, :]
            # except IndexError:
            #     ampls = np.full(240, np.nan)

            if frames.shape[0] == 240:
                val_ampls[val_batch * 240:(val_batch + 1) * 240, :] = find_spot_ampls(frames)

        A0 = (val_ampls[:, ref_row].copy() - Aref[ref_row])[:, np.newaxis]
        z1s = (val_ampls - Aref) * z0_refrow / A0

        z1s = np.delete(z1s, [0, m - 1, ref_row], 1)

        np.save('D:/MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/measured/measured_arr_epoch_{}_raw.npy'
                .format(epoch_num), z1s)

        mask = ~np.isnan(z1s[:, 0])
        z1s = z1s[mask]
        xss = xs[mask]
        yss = ys[mask]

        theories = (xss[:, None] * w1.T).transpose(0, 2, 1).sum(axis=1)

        np.save('D:/MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/measured/measured_arr_epoch_{}.npy'
                .format(epoch_num), z1s)
        np.save('D:/MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL_REF/theory/theory_arr_epoch_{}.npy'
                .format(epoch_num), theories)

        dnn.feedforward(z1s)

        pred = dnn.a2.argmax(axis=1)
        label = yss.argmax(axis=1)

        acc = accuracy(pred, label)
        accs.append(acc)

        np.save('D:/MNIST/accuracy.npy', np.array(accs))

        axs3.plot(accs, linestyle='-', marker='x', c='b')
        plt.draw()
        plt.pause(0.02)
        axs4.plot(loss, linestyle='-', marker='', c='r')
        plt.draw()
        plt.pause(0.02)

        epoch_time = time.time() - t0

        print('\n######################################################################')
        print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                      .format(epoch_num, epoch_time, accs[-1], loss[-1]), 'green'))
        print('######################################################################\n')

        # del xs, ys, frs, target_frames

    print()
    camera.Close()
    imageWindow.Close()
    context.pop()


