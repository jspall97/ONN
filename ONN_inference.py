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
# from make_dmd_image import make_dmd_rgb, make_dmd_image
# from MNIST import map_vec_to_arr
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm1_image import make_slm_rgb, make_dmd_image, make_dmd_batch, update_params
from multiprocessing import Process, Pipe
from ANN import DNN, DNN_1d, accuracy, softmax, accuracy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
from termcolor import colored
import queue
from collections import deque
import ticking
from glumpy.app import clock
from pylon import view_camera

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])

ref_spot = m//2 - 1

ref_block_val = 1.
batch_size = 240
num_batches = 5
num_frames = 10

dmd_block_w = update_params(ref_block_val, batch_size, num_frames)

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

trainX = (trainX * dmd_block_w).astype(int) / dmd_block_w
valX = (valX * dmd_block_w).astype(int) / dmd_block_w
testX = (testX * dmd_block_w).astype(int) / dmd_block_w


testX_cp = cp.array(testX, dtype=cp.float32)


def keyboardinterrupthandler(signal, frame):
    context.pop()
    print('keyboard interupt - closed')
    exit(0)


signal.signal(signal.SIGINT, keyboardinterrupthandler)


def update_slm(arr, lut=False, ref=False):
    global ampl_norm_val

    if arr.shape[1] == m - 1:
        arr = np.insert(arr, ref_spot, np.zeros(n), 1)

    if lut:
        gpu_arr = cp.asarray(arr)
        map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_256 - gpu_arr), axis=0)
        arr_A = cp.linspace(-1., 1., 256)[map_indx].get()

    if ref:
        arr_A[:, ref_spot] = ampl_norm_val

    arr_A = np.flip(arr_A, axis=1)
    img = make_slm_rgb(arr_A, ref_block_val)
    slm.updateArray(img)
    # time.sleep(0.7)


def dmd_one_frame(arr, ref):
    img = make_dmd_image(arr, ref=ref, ref_block_val=ref_block_val)
    return [img]


x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
y_centers = np.load('./tools/y_centers_list.npy')
recomb_params = (35, 619, 84, np.array([602, 637]), np.array([67, 102]), 70)
y_center, mid0, mid1, ref_indxs_0, ref_indxs_1, ref_width = recomb_params


def recombine(arr):
    frame0 = arr[:, :arr.shape[1] // 2].copy()
    frame1 = arr[:, arr.shape[1] // 2:].copy()

    frame1 = np.flip(frame1, axis=1)

    cross0 = frame0[mid0 - (ref_width // 5):mid0 + (ref_width // 5), :].mean(axis=0)
    cross1 = frame1[mid1 - (ref_width // 5):mid1 + (ref_width // 5), :].mean(axis=0)
    #
    # y_center0 = (cross0 * np.arange(cross0.shape[0])).sum() / cross0.sum()
    # y_center1 = (cross1 * np.arange(cross1.shape[0])).sum() / cross1.sum()
    # y_center0 = int(np.round(y_center0, 0))
    # y_center1 = int(np.round(y_center1, 0))

    if arr.max() > 100:

        cross0_bool = (cross0 > cross0.max() * 0.2)
        s0 = cross0_bool.argmax()
        e0 = 70 - np.flip(cross0_bool.copy()).argmax()
        y_center0 = (e0 + s0) / 2

        cross1_bool = (cross1 > cross1.max() * 0.2)
        s1 = cross1_bool.argmax()
        e1 = 70 - np.flip(cross1_bool.copy()).argmax()
        y_center1 = (e1 + s1) / 2

        y_center0 = int(np.round(y_center0, 0))
        y_center1 = int(np.round(y_center1, 0))

    else:

        y_center0 = cross0.argmax()
        y_center1 = cross1.argmax()

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
    arrs = np.array([recombine(arr.T) for arr in arrs_in])

    mask = arrs < 3
    arrs -= 2
    arrs[mask] = 0

    def spot_s(i):
        return np.s_[:, x_edge_indxs[2 * i]:x_edge_indxs[2 * i + 1], y_centers[i] - 1:y_centers[i] + 2]

    spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(m + 1)])

    spot_ampls = cp.sqrt(spot_powers)

    spot_ampls = np.flip(spot_ampls, axis=0)

    ratio = spot_ampls[ref_spot, :] / spot_ampls[ref_spot + 1, :]

    spot_ampls[ref_spot + 1:, :] *= ratio[None, :]

    spot_ampls = np.delete(spot_ampls.get(), ref_spot, 0)

    return spot_ampls.T


actual_uppers_arr_256 = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_256.npy")

actual_uppers_arr_256[:, :, ref_spot] = actual_uppers_arr_256[:, :, ref_spot + 1]

uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
uppers1_ann = np.delete(uppers1_nm, ref_spot, 1)

k = np.abs(np.linspace(-1, 1, 256) - 0.1).argmin()
z0 = actual_uppers_arr_256[k, ...].sum(axis=0)

gpu_actual_uppers_arr_256 = cp.asarray(actual_uppers_arr_256)

if __name__ == '__main__':

    ################
    # SLM display #
    ################

    slm = SLMdisplay()

    ################
    # DMD display #
    ################

    backend = app.use('glfw')

    window = app.Window(1920, 1080, fullscreen=0, decoration=0)
    window.set_position(-1920, 0)
    window.activate()
    window.show()

    dmd_clock = clock.Clock()

    @window.event
    def on_draw(dt):
        global cp_arr, frame_count, target_frames
        window_on_draw(window, screen, cuda_buffer, cp_arr)
        frame_count += 1
        cp_arr = target_frames[frame_count % len(target_frames)]

    screen, cuda_buffer, context = setup(1920, 1080)

    null_frame = dmd_one_frame(np.zeros((n, m)), ref=0)[0]
    null_frames = [null_frame for _ in range(10)]

    full_frame = dmd_one_frame(np.ones((n, m)), ref=0)[0]
    full_frames = [full_frame for _ in range(10)]

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

                self.frames = self.frames[-10000:]

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

    k = np.abs(np.linspace(-1, 1, 256) - ampl_norm_val).argmin()
    slm_arr = actual_uppers_arr_256[k, ...].copy()

    ### Aref ###################

    ref_block_val = 1.
    dmd_block_w = update_params(ref_block_val, batch_size, num_frames)
    update_slm(slm_arr, lut=True, ref=True)
    dmd_arr = np.zeros((n, m))
    target_frames = dmd_one_frame(dmd_arr, ref=1)
    cp_arr = target_frames[0]
    frame_count = 0

    app.__init__(clock=dmd_clock, framerate=0, backend=backend)

    app.run(clock=dmd_clock, framerate=0, framecount=1)
    capture.frames = []
    capture.timestamps = []
    time.sleep(2)
    frs = np.array(capture.frames[-1000:])
    print(frs.shape)
    np.save('./tools/temp_ref.npy', frs)
    Aref = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []

    ### A0 ###################

    ref_block_val = 0.
    dmd_block_w = update_params(ref_block_val, batch_size, num_frames)
    update_slm(slm_arr, lut=True, ref=True)
    dmd_arr = np.ones((n, m))
    target_frames = dmd_one_frame(dmd_arr, ref=1)
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(clock=dmd_clock, framerate=0, framecount=1)
    time.sleep(2)
    frs = np.array(capture.frames[-1000:])
    np.save('./tools/temp_0.npy', frs)
    A0 = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []

    ### Aboth ###################

    ref_block_val = 1.
    dmd_block_w = update_params(ref_block_val, batch_size, num_frames)
    update_slm(slm_arr, lut=True, ref=True)
    dmd_arr = np.ones((n, m))
    target_frames = dmd_one_frame(dmd_arr, ref=1)
    cp_arr = target_frames[0]
    frame_count = 0
    app.run(clock=dmd_clock, framerate=0, framecount=1)
    time.sleep(2)
    frs = np.array(capture.frames[-1000:])
    np.save('./tools/temp_both.npy', frs)
    Aboth = find_spot_ampls(frs).mean(axis=0)
    capture.frames = []
    capture.timestamps = []

    Aref[ref_spot] = Aboth[ref_spot] - A0[ref_spot]

    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))
    axs1.set_ylim(0, 16)
    axs1.plot(Aref, linestyle='', marker='o', c='orange')
    axs1.plot(A0, linestyle='', marker='o', c='g')
    axs1.plot(A0 + Aref, linestyle='', marker='o', c='b')
    axs1.plot(Aboth, linestyle='', marker='x', c='r')
    plt.draw()

    plt.show()

    np.save('./tools/Aref.npy', Aref)

    ######################################

    Aref = np.load('./tools/Aref.npy')
    ampl_norm_val = 0.1

    ref_block_val = 1.
    dmd_block_w = update_params(ref_block_val, batch_size, num_frames)

    batch_size = 240
    num_frames = 10

    w1 = np.load('D:/MNIST/data/best_w1_offline.npy')
    w2 = np.load('D:/MNIST/data/best_w2_offline.npy')

    update_slm(w1, lut=True, ref=True)
    time.sleep(0.7)

    all_z1s = []
    all_theories = []

    target_frames = cp.zeros((16, 1080, 1920, 4), dtype=cp.uint8)
    target_frames[..., -1] = 255
    fc = target_frames.shape[0] - 1
    cp_arr = target_frames[0]
    frame_count = 0
    for _ in range(5):
        capture.frames = []
        app.run(clock=dmd_clock, framerate=0, framecount=fc)
        time.sleep(0.1)

    for k in range(5):

        batch_indxs = np.random.randint(0, 5000, batch_size)
        target_frames[4:-2, :, :, :-1] = make_dmd_batch(testX_cp[batch_indxs, :], 1, ref_block_val, batch_size,
                                                        num_frames)

        xs = testX[batch_indxs, :].copy()

        fc = target_frames.shape[0] - 1
        cp_arr = target_frames[0]
        frame_count = 0

        capture.frames = []
        app.run(clock=dmd_clock, framerate=0, framecount=fc)
        time.sleep(0.1)

        frames = np.array(capture.frames.copy())

        ampls = find_spot_ampls(frames)
        if ampls.shape[0] == 240:
            z1s = ampls - Aref
            z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
            z1s = np.delete(z1s, ref_spot, axis=1)

            theories = np.dot(xs, w1)

            all_z1s.append(z1s)
            all_theories.append(theories)

    all_z1s = np.array(all_z1s)
    all_z1s = all_z1s.reshape(all_z1s.shape[0] * 240, m - 1)
    all_theories = np.array(all_theories)
    all_theories = all_theories.reshape(all_theories.shape[0] * 240, m - 1)

    print(all_z1s.shape, all_theories.shape)

    np.save('./tools/temp_z1s.npy', all_z1s)
    np.save('./tools/temp_theories.npy', all_theories)

    def line(x, grad, c):
        return (grad * x) + c

    norm_params = np.array([curve_fit(line, all_theories[:, j], all_z1s[:, j])[0]
                            for j in range(m - 1)])

    all_z1s -= norm_params[:, 1]
    all_z1s /= norm_params[:, 0]

    print(norm_params)

    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
    axs3.set_ylim(-10, 10)
    axs3.plot([-10, 10], [-10, 10], c='black')
    for j in range(m - 1):
        axs3.plot(all_theories[:, j], all_z1s[:, j], linestyle='', marker='.', markersize=1)
    plt.draw()

    fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
    axs4.set_ylim(-10, 10)
    axs4.plot(all_theories[0, :], linestyle='', marker='o', c='b')
    axs4.plot(all_z1s[0, :], linestyle='', marker='x', c='r')
    plt.draw()

    plt.show()

    # breakpoint()

    ###########
    # TESTING #
    ###########

    start_time = time.time()

    test_z1s = np.full((5000, m - 1), np.nan)

    for test_batch_num in range(21):

        print(test_batch_num)

        if test_batch_num == 20:
            vecs = testX_cp[4800 - 40:, :].copy()
            vecs[:40, :] = 0
        else:
            vecs = testX_cp[test_batch_num * 240:(test_batch_num + 1) * 240, :].copy()

        print(vecs.shape)

        target_frames[4:-2, ..., :-1] = make_dmd_batch(vecs, 1, ref_block_val, batch_size, num_frames)

        fc = target_frames.shape[0] - 1
        cp_arr = target_frames[0]
        frame_count = 0

        capture.frames = []

        app.run(clock=dmd_clock, framerate=0, framecount=fc)

        time.sleep(0.1)

        frames = np.array(capture.frames.copy(), dtype=np.uint8)
        ampls = find_spot_ampls(frames)

        np.save('D:/MNIST/data/testing/images/images_batch_{}.npy'
                .format(test_batch_num), frames)
        np.save('D:/MNIST/data/testing/ampls/ampls_batch_{}.npy'
                .format(test_batch_num), ampls)

        if test_batch_num == 20:
            xs = testX[4800:, :].copy()
        else:
            xs = testX[test_batch_num * 240:(test_batch_num + 1) * 240, :].copy()

        if ampls.shape[0] == batch_size:

            meas = ampls.copy().reshape((num_frames, batch_size // num_frames, m))
            diffs = np.abs(np.array([meas[k + 1, :, m // 3] - meas[k, :, m // 3]
                                     for k in range(num_frames - 1)])).mean(axis=1)
            diffs /= diffs.max()
            repeats = (diffs < 0.25).sum() > 0

            if repeats:
                print(colored('repeated frames, skipping', 'red'))

        else:
            print(colored('wrong num frames: {}'.format(ampls.shape[0]), 'red'))

        if ampls.shape[0] == batch_size and not repeats:

            z1s = ampls - Aref
            z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
            z1s = np.delete(z1s, ref_spot, axis=1)

            z1s -= norm_params[:, 1]
            z1s /= norm_params[:, 0]

            if test_batch_num == 20:
                z1s = z1s[40:, :]
                test_z1s[4800:, :] = z1s.copy()

            else:
                test_z1s[test_batch_num * 240:(test_batch_num + 1) * 240, :] = z1s.copy()

            theories = np.dot(xs, w1.copy())

        else:
            if test_batch_num == 20:
                z1s = np.full((200, m - 1), np.nan)
                theories = np.full((200, m - 1), np.nan)
            else:
                z1s = np.full((batch_size, m - 1), np.nan)
                theories = np.full((batch_size, m - 1), np.nan)

        np.save('D:/MNIST/data/testing/measured/measured_arr_batch_{}.npy'
                .format(test_batch_num), z1s)
        np.save('D:/MNIST/data/testing/theory/theory_arr_batch_{}.npy'
                .format(test_batch_num), theories)

    mask = ~np.isnan(test_z1s[:, 0])
    test_z1s = test_z1s[mask]
    xs = testX.copy()[mask]
    ys = testY.copy()[mask]

    # 1 layer
    # a1s = softmax(test_z1s)
    # pred = a1s.argmax(axis=1)

    def relu(x):
        return np.maximum(0, x)

    # 2 layer
    a1s = relu(test_z1s)
    z2s = np.dot(a1s, w2)
    a2s = softmax(z2s)
    pred = a2s.argmax(axis=1)

    label = ys.argmax(axis=1)

    acc = accuracy(pred, label)

    print(time.time() - start_time)

    print('\n######################################################################')
    print(colored('accuracy : {:.2f}'.format(acc), 'green'))
    print('######################################################################\n')

    print()
    camera.Close()
    # imageWindow.Close()
    context.pop()


