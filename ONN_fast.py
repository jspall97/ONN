from pypylon import pylon
from pypylon import genicam
import time
import sys
import threading
import time
import sys
import signal
import numpy as np
from scipy.optimize import curve_fit
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
from pylon import view_camera

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])
l = int(dims[2])
dmd_block_w = 15
n_nn = 100

inputs = loadmat('C:/Users/spall/OneDrive - Nexus365/Code/JS/controller/onn_test/MNIST digit - subsampled - 100.mat')

num_train = 60000
num_test = 10000

trainY_raw = inputs['trainY']
trainY = np.zeros((num_train, 10))
for i in range(num_train):
    trainY[i, trainY_raw[0, i]] = 1

testY_raw = inputs['testY']
testY = np.zeros((num_test, 10))
for i in range(num_test):
    testY[i, testY_raw[0, i]] = 1

trainX_raw = inputs['trainX']
trainX = np.empty((num_train, 100))
for i in range(num_train):
    trainX_k = trainX_raw[i, :] - trainX_raw[i, :].min()
    trainX_k = trainX_k / trainX_k.max()
    trainX[i, :] = trainX_k

testX_raw = inputs['testX']
testX = np.empty((num_test, 100))
for i in range(num_test):
    testX_k = testX_raw[i, :] - testX_raw[i, :].min()
    testX_k = testX_k / testX_k.max()
    testX[i, :] = testX_k

np.random.seed(0)
np.random.shuffle(trainX)
np.random.seed(0)
np.random.shuffle(trainY)
np.random.seed(0)
np.random.shuffle(testX)
np.random.seed(0)
np.random.shuffle(testY)

valX = testX[:5000, :].copy()
testX = testX[5000:, :].copy()

valY = testY[:5000, :].copy()
testY = testY[5000:, :].copy()

trainX -= 0.1
trainX = np.clip(trainX, 0, 1)
trainX /= trainX.max()

valX -= 0.1
valX = np.clip(valX, 0, 1)
valX /= valX.max()

testX -= 0.1
testX = np.clip(testX, 0, 1)
testX /= testX.max()

trainX = (trainX*dmd_block_w).astype(int)/dmd_block_w
valX = (valX*dmd_block_w).astype(int)/dmd_block_w
testX = (testX*dmd_block_w).astype(int)/dmd_block_w


def keyboardinterrupthandler(signal, frame):
    context.pop()
    print('keyboard interupt - closed')
    exit(0)


signal.signal(signal.SIGINT, keyboardinterrupthandler)

# actual_uppers_arr_1024 = np.load('./tools/actual_uppers_arr_1024.npy')
#
# uppers1_nm = actual_uppers_arr_1024[..., -1].copy()
#
# actual_uppers_arr_1024_T = np.transpose(actual_uppers_arr_1024, (2, 0, 1))
# gpu_actual_uppers_arr_1024 = cp.asarray(actual_uppers_arr_1024_T)

phase_correct_nm = cp.asarray(np.load('./tools/phase_offset.npy'))


def update_slm(arr, lut=False, stag=False, ref=False):

    global ampl_norm_val

    # arr = np.flip(arr, axis=0)

    if arr.shape[1] == m - 1:
        arr = np.insert(arr, 12, np.ones(arr.shape[0]), 1)

    if arr.shape[0] == n_nn:
        arr = np.insert(arr, 50, np.ones((21, arr.shape[1])) * 0.5, 0)

    if lut:
        gpu_arr = cp.asarray(arr)
        map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_1024 - gpu_arr), axis=0)
        arr_A = cp.linspace(-1., 1., 1024)[map_indx]

    else:
        arr_A = cp.asarray(arr)

    if ref:

        arr_A[:, m//2] = ampl_norm_val
        arr_A[50:71, :] = 0.5

    arr_A = cp.flip(arr_A, axis=1)
    arr_phi = cp.angle(arr_A) #- phase_correct_nm.copy()

    img = make_slm_rgb(arr_A, arr_phi, stagger=stag)
    slm.updateArray(img)
    # time.sleep(0.7)


def dmd_one_frame(arr):
    img = make_dmd_image(arr)
    frame = make_dmd_rgb([img for _ in range(24)])
    return [frame]


x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
y_center = 35
mid0 = 603
mid1 = 65
ref_indxs_0 = np.array([581, 625])
ref_indxs_1 = np.array([44, 87])


def recombine(arr):

    frame0 = arr[:, :70].copy()
    frame1 = arr[:, 70:].copy()

    cross0 = frame0[mid0 - 3:mid0 + 3, :].mean(axis=0)
    cross1 = frame1[mid1 - 3:mid1 + 3, :].mean(axis=0)

    crossw = 12
    s0 = cross0.argmax() - crossw
    e0 = cross0.argmax() + crossw + 1
    s1 = cross1.argmax() - crossw
    e1 = cross1.argmax() + crossw + 1

    cross0 = cross0[s0:e0]
    cross1 = cross1[s1:e1]

    y_center0 = (cross0 * np.arange(cross0.shape[0])).sum() / cross0.sum() + s0
    y_center1 = (cross1 * np.arange(cross1.shape[0])).sum() / cross1.sum() + s1

    y_center0 = int(np.round(y_center0, 0))
    y_center1 = int(np.round(y_center1, 0))

    bright_ratio = cross1.mean() / cross0.mean()
    frame0 = frame0 * bright_ratio

    y_delta_0 = y_center0 - y_center
    y_delta_1 = y_center1 - y_center

    edge_cut = np.maximum(np.abs(y_delta_0), np.abs(y_delta_1))

    frame0 = np.roll(frame0, -y_delta_0, axis=1)
    frame1 = np.roll(frame1, -y_delta_1, axis=1)

    frame0 = frame0[:ref_indxs_0[1], :]
    frame1 = frame1[ref_indxs_1[0]:, :]
    frame_both = np.concatenate((frame0, frame1))

    frame_both[:, :edge_cut] = 0
    frame_both[:, -edge_cut:] = 0

    return frame_both


def find_spot_ampls(arrs_in):

    arrs = [recombine(arr_in.T) for arr_in in arrs_in]

    all_spot_ampls = []

    for arr in arrs:

        mask = arr < 3
        arr -= 2
        arr[mask] = 0

        def spot_s(ii):
            return np.s_[x_edge_indxs[2 * ii]:x_edge_indxs[2 * ii + 1], y_center - 1:y_center + 2]

        spots_dict = {}

        for spot_num in range(m+1):
            spot = arr[spot_s(spot_num)]

            spots_dict[spot_num] = spot

        spot_powers = np.array([spots_dict[ii].mean() for ii in range(m+1)])

        spot_ampls = np.sqrt(spot_powers)

        spot_ampls[m // 2 + 1:] *= spot_ampls[m // 2] / spot_ampls[m // 2 + 1]
        spot_ampls = np.delete(spot_ampls, 13)

        all_spot_ampls.append(np.flip(spot_ampls))

    return np.array(all_spot_ampls)


# def find_spot_ampls(frames):
#
#     cp_frames = cp.array(frames, dtype=cp.float16)
#
#     frame0 = cp_frames[:, :, :70]  # .copy()
#     frame1 = cp_frames[:, :, 70:]  # .copy()
#
#     cross0 = frame0[:, mid0 - 3:mid0 + 3, :].mean(axis=1)
#     cross1 = frame1[:, mid1 - 3:mid1 + 3, :].mean(axis=1)
#
#     y_center0 = ((cross0 * cp.arange(cross0.shape[1])).sum(axis=1)) / cross0.sum(axis=1)
#     y_center1 = ((cross1 * cp.arange(cross1.shape[1])).sum(axis=1)) / cross1.sum(axis=1)
#
#     y_center0 = int(cp.around(y_center0.mean(), 0))
#     y_center1 = int(cp.around(y_center1.mean(), 0))
#
#     bright_ratio = cross1.mean(axis=1) / cross0.mean(axis=1)
#
#     frame0 *= bright_ratio.mean()
#
#     y_delta_0 = y_center0 - y_center
#     y_delta_1 = y_center1 - y_center
#
#     edge_cut = int(cp.maximum(cp.abs(y_delta_0), cp.abs(y_delta_1)).max())
#
#     frame0 = cp.roll(frame0, -y_delta_0, axis=2).astype(cp.uint8)
#     frame1 = cp.roll(frame1, -y_delta_1, axis=2).astype(cp.uint8)
#
#     frame0 = frame0[:, :ref_indxs_0[1], :]
#     frame1 = frame1[:, ref_indxs_1[0]:, :]
#
#     frame_both = cp.empty((frame0.shape[0], frame0.shape[1] + frame1.shape[1], 70), dtype=cp.uint8)
#     frame_both[:, :frame0.shape[1], :] = frame0
#     frame_both[:, frame0.shape[1]:, :] = frame1
#
#     frame_both[:, :, :edge_cut] = 0
#     frame_both[:, :, -edge_cut:] = 0
#
#     mask = frame_both < 3
#     frame_both -= 2
#     frame_both[mask] = 0
#
#     def spot_s(i):
#         return np.s_[:, x_edge_indxs[2 * i]:x_edge_indxs[2 * i + 1], y_center - 1:y_center + 2]
#
#     spot_powers = cp.array([frame_both[spot_s(i)].mean(axis=(1, 2)) for i in range(m + 1)]).T
#
#     spot_ampls = cp.sqrt(spot_powers)
#
#     ratio = spot_ampls[:, m // 2] / spot_ampls[:, m // 2 + 1]
#
#     spot_ampls = spot_ampls.T
#     spot_ampls[m // 2 + 1:, :] *= ratio
#     spot_ampls = spot_ampls.T
#
#     spot_ampls = np.delete(spot_ampls.get(), 13, 1)
#     spot_ampls = np.flip(spot_ampls, axis=1)
#
#     del frame0, frame1, cross0, cross1, frame_both, spot_powers, mask, cp_frames
#
#     return spot_ampls


actual_uppers_arr_1024 = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_1024.npy")

ref_spot = m//2

actual_uppers_arr_1024[:, 50:71, :] = 0

actual_uppers_arr_1024[:, :, 0] = actual_uppers_arr_1024[:, :, 1]

actual_uppers_arr_1024 = actual_uppers_arr_1024/actual_uppers_arr_1024.max()
actual_uppers_arr_1024[:, :, ref_spot] = actual_uppers_arr_1024[:, :, ref_spot+1]

uppers1_nm = actual_uppers_arr_1024[-1, ...].copy()
uppers1_ann = np.delete(uppers1_nm, np.arange(50, 71), 0)
uppers1_ann = np.delete(uppers1_ann, ref_spot, 1)

k = np.abs(np.linspace(-1, 1, 1024) - 0.1).argmin()
z0 = actual_uppers_arr_1024[k, ...].sum(axis=0)

gpu_actual_uppers_arr_1024 = cp.asarray(actual_uppers_arr_1024)


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

    null_frame = dmd_one_frame(np.zeros((n, m)))[0]
    null_frames = [null_frame for _ in range(10)]

    full_frame = dmd_one_frame(np.ones((n, m)))[0]
    full_frames = [full_frame for _ in range(10)]

    ###################
    # set brightness #
    ###################

    # ampl_norm_val = 0.1
    #
    # slm_arr = uppers1_nm.copy() * ampl_norm_val
    # # slm_arr[:, ::2] = 0
    # # slm_arr[-40:, -5:] = 0
    # update_slm(slm_arr, lut=True, ref=True)
    #
    # dmd_arr = np.zeros((n, m))
    # dmd_arr[:, m // 2] = 1
    # dmd_arr[50:n - 50, :] = 1
    # target_frames = dmd_one_frame(dmd_arr)
    # cp_arr = target_frames[0]
    # frame_count = 0
    # app.run(framerate=0, framecount=1)
    #
    # view_camera()
    #
    # dmd_arr = np.ones((n, m))
    # dmd_arr[50:n - 50, :] = 0
    # target_frames = dmd_one_frame(dmd_arr)
    # cp_arr = target_frames[0]
    # frame_count = 0
    # app.run(framerate=0, framecount=1)
    #
    # view_camera()
    #
    # dmd_arr = np.ones((n, m))
    # target_frames = dmd_one_frame(dmd_arr)
    # cp_arr = target_frames[0]
    # frame_count = 0
    # app.run(framerate=0, framecount=1)
    #
    # view_camera()
    #
    # slm_arr = uppers1_nm.copy() * -1. * ampl_norm_val
    # update_slm(slm_arr, lut=True, ref=True)
    #
    # view_camera()

    ################
    # Pylon camera #
    ################

    # imageWindow = pylon.PylonImageWindow()
    # imageWindow.Create(1)

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

            self.frames = []

        def OnImageGrabbed(self, cam, grab_result):
            if grab_result.GrabSucceeded():

                image = grab_result.GetArray()

                if image.max() > 10:
                    self.frames.append(image)

                self.frames = self.frames[-5000:]

                # imageWindow.SetImage(grab_result)
                # imageWindow.Show()

    # register the background handler and start grabbing using background pylon thread
    capture = CaptureProcess()
    camera.RegisterImageEventHandler(capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
    time.sleep(1)

    #####################
    # find norm values #
    #####################

    ampl_norm_val = 0.1

    k = np.abs(np.linspace(-1, 1, 1024) - ampl_norm_val).argmin()
    slm_arr = actual_uppers_arr_1024[k, ...].copy()
    update_slm(slm_arr, lut=True, ref=True)
    time.sleep(1)

    dmd_arr = np.zeros((n, m))
    dmd_arr[:, ref_spot] = 1
    dmd_arr[50:71, :] = 1
    target_frames = dmd_one_frame(dmd_arr)
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(framerate=0, framecount=1)

    capture.frames = []
    capture.timestamps = []

    time.sleep(2)
    frs = np.array(capture.frames[-1000:])
    print(frs.shape)
    np.save('./tools/temp_ref.npy', frs)
    Aref = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []

    dmd_arr = np.ones((n, m))
    dmd_arr[50:71, :] = 0
    target_frames = dmd_one_frame(dmd_arr)
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(framerate=0, framecount=1)

    time.sleep(2)
    frs = np.array(capture.frames[-1000:])
    np.save('./tools/temp_0.npy', frs)
    A0 = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []

    dmd_arr = np.ones((n, m))
    target_frames = dmd_one_frame(dmd_arr)
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(framerate=0, framecount=1)

    time.sleep(2)
    frs = np.array(capture.frames[-1000:])
    np.save('./tools/temp_both.npy', frs)
    Aboth = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []

    Aref[ref_spot] = Aboth[ref_spot] - A0[ref_spot]

    # print(Aref, A0, Aboth)

    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))
    axs1.set_ylim(0, 16)
    axs1.plot(Aref, linestyle='', marker='o', c='orange')
    axs1.plot(A0, linestyle='', marker='o', c='g')
    axs1.plot(A0 + Aref, linestyle='', marker='o', c='b')
    axs1.plot(Aboth, linestyle='', marker='x', c='r')
    plt.draw()

    plt.show()

    ######################################

    np.random.seed(0)
    slm_arr = np.random.normal(0, 0.5, (n_nn, m-1))
    slm_arr = np.clip(slm_arr, -uppers1_ann, uppers1_ann)
    update_slm(slm_arr, lut=True, ref=True)
    time.sleep(1)

    xs = trainX[10, :].copy()
    dmd_arr = xs[:, np.newaxis].repeat(m-1, axis=1)

    dmd_disp_arr = np.insert(dmd_arr, 50, np.ones((21, m-1)), 0)
    dmd_disp_arr = np.insert(dmd_disp_arr, ref_spot, np.ones(n), 1)

    target_frames = dmd_one_frame(dmd_disp_arr)
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(framerate=0, framecount=1)
    time.sleep(2)

    z1_theory = (slm_arr*np.flip(dmd_arr, axis=0)).sum(axis=0)

    frs = np.array(capture.frames[-1000:])
    Ameas = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []
    z1 = Ameas - Aref
    z1 *= z0[ref_spot] / z1[ref_spot]
    z1 = np.delete(z1, ref_spot)

    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 4))
    axs2.set_ylim(-10, 10)
    axs2.plot(z1_theory, linestyle='', marker='o', c='b')
    axs2.plot(z1, linestyle='', marker='x', c='r')
    plt.draw()

    # input()

    # breakpoint()

    print()
    print('############')
    print()

    num_frames = 10
    batch_indxs = np.arange(num_frames)

    xs = np.load('D:/MNIST/trainX/xs/xs_{}.npy'.format(0))
    ys = np.load('D:/MNIST/trainX/ys/ys_{}.npy'.format(0))
    for j in range(1, num_frames):
        xs = np.concatenate((xs, np.load('D:/MNIST/trainX/xs/xs_{}.npy'.format(j))))
        ys = np.concatenate((ys, np.load('D:/MNIST/trainX/ys/ys_{}.npy'.format(j))))
    xs = np.delete(xs, np.arange(50, 71), axis=1)
    xs = np.delete(xs, ref_spot, axis=2)

    w1 = slm_arr.copy()

    target_frames = [cp.load('./MNIST/trainX_rgb_frames_m24_no_invert/rgb24_{}.npy'.format(j))[0, ...]
                     for j in batch_indxs]
    for _ in range(2):
        target_frames[0:0] = [null_frame]
        target_frames.extend([null_frame])

    fc = len(target_frames) - 1
    cp_arr = target_frames[0]
    frame_count = 0

    for _ in range(5):
        capture.frames = []
        app.run(framerate=0, framecount=fc)
        time.sleep(0.1)

    capture.frames = []
    app.run(framerate=0, framecount=fc)
    time.sleep(0.1)

    frames = np.array(capture.frames.copy())
    print(frames.shape)
    # np.save('D:/MNIST/pylon_captures/training/frames/batch_{}.npy'.format(0), frames)

    # frames = np.load('D:/MNIST/pylon_captures/frames/batch_{}.npy'.format(0))

    ampls = find_spot_ampls(frames)
    print(ampls.shape)
    # np.save('D:/MNIST/pylon_captures/training/ampls/batch_{}.npy'.format(0), ampls)

    assert ampls.shape[0] == 240

    z1s = ampls - Aref
    z1s = (z1s.T * z0[ref_spot] / z1s[:, ref_spot]).T
    z1s = np.delete(z1s, ref_spot, axis=1)

    theories = (xs * w1).sum(axis=1)

    print(z1s.shape, theories.shape)

    def line(x, grad):
        return grad * x

    norm_params = np.array([curve_fit(line, theories[:, j], z1s[:, j])[0] for j in range(m - 1)])

    # z1s -= norm_params[:, 1]
    z1s /= norm_params[:, 0]

    print(norm_params.shape)

    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
    axs3.set_ylim(-10, 10)
    axs3.plot([-10, 10], [-10, 10], c='black')
    for j in range(m-1):
        axs3.plot(theories[:, j], z1s[:, j], linestyle='', marker='.', markersize=1)
    plt.draw()

    fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
    axs4.set_ylim(-10, 10)
    axs4.plot(theories[0, :], linestyle='', marker='o', c='b')
    axs4.plot(z1s[0, :], linestyle='', marker='x', c='r')
    plt.draw()

    plt.show()

    # breakpoint()

    ###########################
    # loop batches and epochs #
    ###########################

    num_batches = 50
    num_frames = 10
    batch_size = num_frames * 24
    num_epochs = 10

    n_nn = 100

    lim_arr = uppers1_ann.copy()

    w1 = np.random.normal(0, 0.5, (n_nn, m-1))
    w1 = np.clip(w1, -lim_arr, lim_arr)

    update_slm(w1, lut=True, stag=False, ref=True)
    time.sleep(1)

    best_w1 = w1.copy()

    w2 = np.random.normal(0, 0.5, (m-1, l))

    m_dw1 = np.zeros((n_nn, m-1))
    v_dw1 = np.zeros((n_nn, m-1))
    m_dw2 = np.zeros((m-1, l))
    v_dw2 = np.zeros((m-1, l))
    beta1 = 0.9
    beta2 = 0.999
    adam_params = (m_dw1, v_dw1, m_dw2, v_dw2, beta1, beta2)

    # w1 = np.load('./MNIST/testX/w1.npy')
    # w2 = np.load('./MNIST/testX/w2.npy')

    # w1 = np.load('./MNIST/w1/w1_epoch_{}_batch_{}.npy'.format(3, 49))
    # w2 = np.load('./MNIST/w2/w2_epoch_{}_batch_{}.npy'.format(3, 49))
    # adam_params = list(np.load('./MNIST/adam_params.npy', allow_pickle=True))
    # update_slm(w1, lut=True)
    # time.sleep(1)

    dnn = DNN(*adam_params, x=trainX, y=trainY, w1_0=w1, w2_0=w2, batch_size=batch_size, num_batches=num_batches,
              lr=5e-3)

    loss = [5]
    accs = []

    fig3, [[axs0, axs1], [axs2, axs3]] = plt.subplots(2, 2, figsize=(8, 4))

    axs0.set_ylim(-10, 10)
    axs1.set_ylim(-10, 10)
    axs1.set_xlim(-10, 10)

    axs1.plot([-10, 10], [-10, 10], c='black')
    eg_line = [axs1.plot(theories[:, j], z1s[:, j], linestyle='', marker='.', markersize=1)[0] for j in range(m - 1)]

    th_line = axs0.plot(theories[0, :], linestyle='', marker='o', c='b')[0]
    meas_line = axs0.plot(z1s[0, :], linestyle='', marker='x', c='r')[0]

    axs2.set_ylim(0, 100)
    axs2.set_xlim(0, 30)
    axs2.plot(accs, linestyle='-', marker='x', c='b')

    axs3.set_ylim(0, 5)
    axs3.set_xlim(0, 1500)
    axs3.plot(loss, linestyle='-', marker='', c='r')

    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.001)

    loop_clock = clock.Clock()
    loop_clock.tick()

    for epoch_num in range(num_epochs):

        epoch_rand_indxs = np.arange(60000 // 24)
        random.Random(epoch_num).shuffle(epoch_rand_indxs)

        batch_indxs = []
        for i in range(num_batches):
            batch_indxs.append(epoch_rand_indxs[i * num_frames: (i + 1) * num_frames])

        epoch_xs = []
        epoch_ys = []

        for i in range(num_batches):

            for j, indx in enumerate(batch_indxs[i]):
                if j == 0:
                    xs_i = np.load('D:/MNIST/trainX/xs/xs_{}.npy'.format(indx))
                    ys_i = np.load('D:/MNIST/trainX/ys/ys_{}.npy'.format(indx))
                else:
                    xs_i = np.concatenate((xs_i, np.load('D:/MNIST/trainX/xs/xs_{}.npy'.format(indx))))
                    ys_i = np.concatenate((ys_i, np.load('D:/MNIST/trainX/ys/ys_{}.npy'.format(indx))))
            xs_i = np.delete(xs_i, np.arange(50, 71), axis=1)
            xs_i = np.delete(xs_i, ref_spot, axis=2)
            epoch_xs.append(xs_i)
            epoch_ys.append(ys_i)

        # init loop
        target_frames = full_frames.copy()
        target_frames[0:0] = [null_frame]
        target_frames.extend([null_frame])
        fc = len(target_frames)-1

        cp_arr = target_frames[0]
        frame_count = 0

        for _ in range(5):
            capture.frames = []
            capture.timestamps = []

            app.run(framerate=0, framecount=fc)

            time.sleep(0.1)
            loop_clock.tick()

        print()

        t0 = time.time()

        for batch_num in range(num_batches):

            print(batch_num)

            target_frames = [cp.load('./MNIST/trainX_rgb_frames_m24_no_invert/rgb24_{}.npy'.format(j))[0, ...]
                             for j in batch_indxs[batch_num]]

            for _ in range(2):
                target_frames[0:0] = [null_frame]
                target_frames.extend([null_frame])

            fc = len(target_frames)-1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []

            app.run(framerate=0, framecount=fc)

            time.sleep(0.1)

            frames = np.array(capture.frames.copy())

            ampls = find_spot_ampls(frames)

            print(frames.max())
            print((ampls**2).max())

            np.save('D:/MNIST/data/training/images/images_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), frames)
            np.save('D:/MNIST/data/training/ampls/ampls_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), ampls)

            if ampls.shape[0] == batch_size:

                z1s = ampls - Aref
                z1s = (z1s.T * z0[ref_spot] / z1s[:, ref_spot]).T
                z1s = np.delete(z1s, ref_spot, axis=1)

                # z1s -= norm_params[:, 1]
                z1s /= norm_params[:, 0]

                xs = epoch_xs[batch_num]
                ys = epoch_ys[batch_num]

                theories = (xs * w1).sum(axis=1)

                dnn.feedforward(z1s)

                dnn.backprop(xs[:, :, 0], ys)

                dnn.w1 = np.clip(dnn.w1, -uppers1_ann, uppers1_ann)

                w1 = dnn.w1.copy()
                update_slm(w1, lut=True, ref=True)
                # time.sleep(0.7)

                if dnn.loss < loss[-1]:
                    best_w1 = w1.copy()

                loss.append(dnn.loss)
                print(colored('loss : {:.2f}'.format(dnn.loss), 'green'))

                np.save('D:/MNIST/data/loss.npy', np.array(loss))
                new_adam_params = np.array([dnn.m_dw1, dnn.v_dw1, dnn.m_dw2, dnn.v_dw2, dnn.beta1, dnn.beta2])
                np.save('D:/MNIST/data/adam_params.npy', new_adam_params)

            else:
                print(colored('wrong num frames: {}'.format(ampls.shape[0]), 'red'))
                z1s = np.full((batch_size, m-1), np.nan)
                theories = np.full((batch_size, m-1), np.nan)

            np.save('D:/MNIST/data/training/measured/measured_arr_epoch_{}_batch_{}.npy'
                    .format(epoch_num, batch_num), z1s)
            np.save('D:/MNIST/data/training/theory/theory_arr_epoch_{}_batch_{}.npy'
                    .format(epoch_num, batch_num), theories)

            np.save('D:/MNIST/data/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1))
            np.save('D:/MNIST/data/w2/w2_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w2))

            for j in range(m - 1):
                eg_line[j].set_xdata(theories[:, j])
                eg_line[j].set_ydata(z1s[:, j])

            th_line.set_ydata(theories[0, :])
            meas_line.set_ydata(z1s[0, :])

            plt.draw()
            # plt.pause(0.001)
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.001)

            dt = loop_clock.tick()
            print(dt)
            print()

        ##################
        ### VALIDATION ###
        ##################

        dnn.w1 = best_w1.copy()
        update_slm(dnn.w1, lut=True, ref=True)
        time.sleep(0.7)

        ampls = np.full((4800, m), np.nan)

        for val_batch in range(20):

            frs = cp.load('D:/MNIST/valX/frames/rgb24_{}.npy'.format(val_batch))
            target_frames = [frs[i, ...] for i in range(10)]

            for _ in range(2):
                target_frames[0:0] = [null_frame]
                target_frames.extend([null_frame])

            fc = len(target_frames) - 1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []
            capture.timestamps = []

            app.run(framerate=0, framecount=fc)

            time.sleep(0.1)

            frames = np.array(capture.frames.copy())

            np.save('D:/MNIST/data/validation/images/epoch_{}_batch_{}.npy'.format(epoch_num, val_batch), frames)

            if frames.shape[0] == 240:
                ampls[val_batch * 240:(val_batch + 1) * 240, :] = find_spot_ampls(frames)
            else:
                print(colored('wrong num frames, skipping', 'red'))

            np.save('D:/MNIST/data/validation/ampls/epoch_{}_batch_{}.npy'.format(epoch_num, val_batch), frames)

        z1s = ampls - Aref
        z1s = (z1s.T * z0[ref_spot] / z1s[:, ref_spot]).T
        z1s = np.delete(z1s, ref_spot, axis=1)
        # z1s -= norm_params[:, 1]
        z1s /= norm_params[:, 0]

        np.save('D:/MNIST/data/validation/measured/measured_arr_epoch_{}_raw.npy'
                .format(epoch_num), z1s)

        xs = np.load('D:/MNIST/valX/xs.npy')
        xs = np.delete(xs, np.arange(50, 71), axis=1)
        xs = np.delete(xs, ref_spot, axis=2)
        ys = np.load('D:/MNIST/valX/ys.npy')

        mask = ~np.isnan(z1s[:, 0])
        z1s = z1s[mask]
        xss = xs[mask]
        yss = ys[mask]

        theories = (xss * w1).sum(axis=1)

        np.save('D:/MNIST/data/validation/measured/measured_arr_epoch_{}.npy'
                .format(epoch_num), z1s)
        np.save('D:/MNIST/data/validation/theory/theory_arr_epoch_{}_raw.npy'
                .format(epoch_num), theories)

        dnn.feedforward(z1s)

        pred = dnn.a2.argmax(axis=1)
        label = yss.argmax(axis=1)

        acc = accuracy(pred, label)
        accs.append(acc)

        np.save('D:/MNIST/data/accuracy.npy', np.array(accs))

        axs2.plot(accs, linestyle='-', marker='x', c='b')
        axs3.plot(loss, linestyle='-', marker='', c='r')
        plt.draw()
        plt.pause(0.001)

        epoch_time = time.time() - t0

        print('\n######################################################################')
        print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                      .format(epoch_num, epoch_time, accs[-1], loss[-1]), 'green'))
        print('######################################################################\n')


    print()
    camera.Close()
    # imageWindow.Close()
    context.pop()


