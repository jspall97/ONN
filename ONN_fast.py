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
from ANN import DNN, DNN_1d, DNN_complex, accuracy
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

ref_spot = m//2

ref_block_val = 0.3
batch_size = 240
num_batches = 5
num_frames = 10

dmd_block_w = update_params(ref_block_val, batch_size, num_frames, is_complex=True)


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

trainX_cp = cp.array(trainX, dtype=cp.float32)
valX_cp = cp.array(valX, dtype=cp.float32)
testX_cp = cp.array(testX, dtype=cp.float32)


def keyboardinterrupthandler(signal, frame):
    context.pop()
    print('keyboard interupt - closed')
    exit(0)


signal.signal(signal.SIGINT, keyboardinterrupthandler)

complex_output_ratios = np.load('./tools/complex_output_ratios.npy')


def update_slm(arr, lut=False, ref=False, noise_arr_A=None, noise_arr_phi=None):

    global ampl_norm_val, ref_spot

    if arr.shape[1] == m-1:
        arr = np.insert(arr, ref_spot, np.zeros(n), 1)

    if arr.shape[1] == 10:
        arr = np.repeat(arr.copy(), 4, axis=1) * complex_output_ratios.copy()[None, :]

    if lut:
        # gpu_arr = cp.asarray(arr)
        # map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_256 - gpu_arr), axis=0)
        # arr_A = cp.linspace(-1., 1., 256)[map_indx].get()

        gpu_arr = cp.abs(cp.asarray(arr.copy()))
        map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_128_flat - gpu_arr), axis=0)
        arr_A = cp.linspace(0, 1, 128)[map_indx]

    else:
        arr_A = cp.abs(cp.asarray(arr.copy()))

    if ref:
        arr_A[:, ref_spot] = ampl_norm_val

    arr_phi = cp.angle(cp.array(arr.copy()))

    if noise_arr_A is not None:
        arr_A += cp.array(noise_arr_A)

    if noise_arr_phi is not None:
        arr_phi += cp.array(noise_arr_phi)

    arr_out = arr_A * cp.exp(1j*arr_phi)

    arr_out = np.flip(arr_out.get(), axis=1)
    img = make_slm_rgb(arr_out, ref_block_val)
    slm.updateArray(img)
    # time.sleep(0.7)


def dmd_one_frame(arr, ref):
    img = make_dmd_image(arr, ref=ref, ref_block_val=ref_block_val)
    return [img]


# recomb_params = (35, 603, 68, np.array([578, 628]), np.array([43, 93]), 100)
# y_center, mid0, mid1, ref_indxs_0, ref_indxs_1, ref_width = recomb_params


# def recombine(arr):
#
#     frame0 = arr[:, :arr.shape[1] // 2].copy()
#     frame1 = arr[:, arr.shape[1] // 2:].copy()
#
#     frame1 = np.flip(frame1, axis=1)
#
#     cross0 = frame0[mid0-(ref_width//5):mid0+(ref_width//5), :].mean(axis=0)
#     cross1 = frame1[mid1-(ref_width//5):mid1+(ref_width//5), :].mean(axis=0)
#     #
#     # y_center0 = (cross0 * np.arange(cross0.shape[0])).sum() / cross0.sum()
#     # y_center1 = (cross1 * np.arange(cross1.shape[0])).sum() / cross1.sum()
#     # y_center0 = int(np.round(y_center0, 0))
#     # y_center1 = int(np.round(y_center1, 0))
#
#     # if arr.max() > 100:
#     #
#     #     cross0_bool = (cross0 > cross0.max() * 0.2)
#     #     s0 = cross0_bool.argmax()
#     #     e0 = 70 - np.flip(cross0_bool.copy()).argmax()
#     #     y_center0 = (e0 + s0) / 2
#     #
#     #     cross1_bool = (cross1 > cross1.max() * 0.2)
#     #     s1 = cross1_bool.argmax()
#     #     e1 = 70 - np.flip(cross1_bool.copy()).argmax()
#     #     y_center1 = (e1 + s1) / 2
#     #
#     #     y_center0 = int(np.round(y_center0, 0))
#     #     y_center1 = int(np.round(y_center1, 0))
#     #
#     # else:
#     #
#     #     y_center0 = cross0.argmax()
#     #     y_center1 = cross1.argmax()
#
#     bright_ratio = cross1.mean() / cross0.mean()
#     frame0 = frame0 * bright_ratio
#
#     # y_delta_0 = y_center0 - y_center
#     # y_delta_1 = y_center1 - y_center
#
#     y_delta_0 = 0 #7
#     y_delta_1 = 1 #0
#
#     edge_cut = np.maximum(np.abs(y_delta_0), np.abs(y_delta_1))
#
#     frame0 = np.roll(frame0, -y_delta_0, axis=1)
#     frame1 = np.roll(frame1, -y_delta_1, axis=1)
#
#     frame0 = frame0[:ref_indxs_0[1], :]
#     frame1 = frame1[ref_indxs_1[0]:, :]
#     frame_both = np.concatenate((frame0, frame1))
#
#     frame_both[:, :edge_cut] = 0
#     frame_both[:, -edge_cut:] = 0
#
#     return frame_both

x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
y_centers = np.load('./tools/y_centers_list.npy')


def find_spot_ampls(arrs):

    # arrs = np.array([recombine(arr.T) for arr in arrs_in])

    mask = arrs < 3
    arrs -= 2
    arrs[mask] = 0

    def spot_s(i):
        return np.s_[:, y_centers[i] - 2:y_centers[i] + 3, x_edge_indxs[2 * i]:x_edge_indxs[2 * i + 1]]

    spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(m + 1)])

    spot_ampls = cp.sqrt(spot_powers)

    spot_ampls = np.flip(spot_ampls, axis=0)

    ratio = spot_ampls[ref_spot, :] / spot_ampls[ref_spot + 1, :]

    spot_ampls[ref_spot + 1:, :] *= ratio[None, :]

    spot_ampls = np.delete(spot_ampls.get(), ref_spot+1, 0)

    return spot_ampls.T


# actual_uppers_arr_256 = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_256.npy")
#
# actual_uppers_arr_256[:, :, ref_spot] = actual_uppers_arr_256[:, :, ref_spot+1]
#
# uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
# uppers1_ann = np.delete(uppers1_nm, ref_spot, 1)
#
# k = np.abs(np.linspace(-1, 1, 256) - 0.1).argmin()
# z0 = actual_uppers_arr_256[k, ...].sum(axis=0)
#
# z0_norm = z0.copy()/z0.max()
# z0_norm = np.delete(z0_norm, ref_spot)
#
# gpu_actual_uppers_arr_256 = cp.asarray(actual_uppers_arr_256)


actual_uppers_arr_128_flat = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_128_flat.npy")

actual_uppers_arr_128_flat /= actual_uppers_arr_128_flat.max()

uppers1_nm_flat = actual_uppers_arr_128_flat[-1, ...].copy()

uppers1_ann = uppers1_nm_flat.copy()[:, ::4]

gpu_actual_uppers_arr_128_flat = cp.asarray(actual_uppers_arr_128_flat)


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
    window.set_position(-1920*2, 0)
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

                self.frames = self.frames[-1001:]

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

    # ampl_norm_val = 0.1
    #
    # k = np.abs(np.linspace(-1, 1, 256) - ampl_norm_val).argmin()
    # slm_arr = actual_uppers_arr_256[k, ...].copy()

    # # ### Aref ###################
    # #
    # ref_block_val = 1.
    # dmd_block_w = update_params(ref_block_val, batch_size, num_frames)
    # update_slm(slm_arr, lut=True, ref=True)
    # dmd_arr = np.zeros((n, m))
    # target_frames = dmd_one_frame(dmd_arr, ref=1)
    # cp_arr = target_frames[0]
    # frame_count = 0
    #
    # app.__init__(clock=dmd_clock, framerate=0, backend=backend)
    #
    # app.run(clock=dmd_clock, framerate=0, framecount=1)
    # capture.frames = []
    # capture.timestamps = []
    # time.sleep(1)
    # frs = np.array(capture.frames[-1000:])
    # np.save('./tools/temp_ref.npy', frs)
    # Aref = find_spot_ampls(frs).mean(axis=0)
    # capture.frames = []
    # capture.timestamps = []
    #
    # ### A0 ###################
    #
    # ref_block_val = 0.
    # dmd_block_w = update_params(ref_block_val, batch_size, num_frames)
    # update_slm(slm_arr, lut=True, ref=True)
    # dmd_arr = np.ones((n, m))
    # target_frames = dmd_one_frame(dmd_arr, ref=1)
    # cp_arr = target_frames[0]
    # frame_count = 0
    # app.run(clock=dmd_clock, framerate=0, framecount=1)
    # time.sleep(1)
    # frs = np.array(capture.frames[-1000:])
    # np.save('./tools/temp_0.npy', frs)
    # A0 = find_spot_ampls(frs).mean(axis=0)
    # capture.frames = []
    # capture.timestamps = []
    #
    # ### Aboth ###################
    #
    # ref_block_val = 1.
    # dmd_block_w = update_params(ref_block_val, batch_size, num_frames)
    # update_slm(slm_arr, lut=True, ref=True)
    # dmd_arr = np.ones((n, m))
    # target_frames = dmd_one_frame(dmd_arr, ref=1)
    # cp_arr = target_frames[0]
    # frame_count = 0
    # app.run(clock=dmd_clock, framerate=0, framecount=1)
    # time.sleep(1)
    # frs = np.array(capture.frames[-1000:])
    # np.save('./tools/temp_both.npy', frs)
    # Aboth = find_spot_ampls(frs).mean(axis=0)
    # capture.frames = []
    # capture.timestamps = []
    #
    # Aref[ref_spot] = Aboth[ref_spot] - A0[ref_spot]
    #
    # fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))
    # axs1.set_ylim(0, 16)
    # axs1.plot(Aref, linestyle='', marker='o', c='orange')
    # axs1.plot(A0, linestyle='', marker='o', c='g')
    # axs1.plot(A0 + Aref, linestyle='', marker='o', c='b')
    # axs1.plot(Aboth, linestyle='', marker='x', c='r')
    # plt.draw()
    #
    # plt.show()
    #
    # np.save('./tools/Aref.npy', Aref)

    print('############')

    ampl_norm_val = 0.1
    # Aref = np.load('./tools/Aref.npy')

    scale_guess = 10

    batch_size = 240
    num_frames = 10

    all_z1s = []
    all_theories = []

    target_frames = cp.zeros((14, 1080, 1920, 4), dtype=cp.uint8)
    target_frames[..., -1] = 255
    fc = target_frames.shape[0] - 1
    cp_arr = target_frames[0]
    frame_count = 0
    for _ in range(5):
        capture.frames = []
        app.run(clock=dmd_clock, framerate=0, framecount=fc)
        time.sleep(0.1)

    for k in range(5):

        # np.random.seed(k)
        # slm_arr = np.random.normal(0, 0.5, (n, m))
        # slm_arr = np.clip(slm_arr, -uppers1_nm, uppers1_nm)
        # update_slm(slm_arr, lut=True, ref=True)
        # time.sleep(1)

        np.random.seed(k)

        slm_arr_X = np.random.normal(0, 0.5, (n, 10))
        slm_arr_X = np.clip(slm_arr_X, -uppers1_ann, uppers1_ann)

        slm_arr_Y = np.random.normal(0, 0.5, (n, 10))
        slm_arr_Y = np.clip(slm_arr_Y, -uppers1_ann, uppers1_ann)

        slm_arr_Z = slm_arr_X + (1j*slm_arr_Y)

        # CHECK REFERENCE IS ON/OFF
        update_slm(slm_arr_Z, lut=True, ref=False)
        time.sleep(1)

        batch_indxs = np.random.randint(0, 60000, batch_size)
        # CHECK REFERENCE IS ON/OFF
        target_frames[2:-2, :, :, :-1] = make_dmd_batch(trainX_cp[batch_indxs, :], 0,
                                                        ref_block_val, batch_size, num_frames)

        xs = trainX[batch_indxs, :].copy()

        cp_arr = target_frames[0]
        frame_count = 0

        capture.frames = []
        app.run(clock=dmd_clock, framerate=0, framecount=fc)
        time.sleep(0.1)

        frames = np.array(capture.frames.copy())

        print(frames.shape)

        ampls = find_spot_ampls(frames)

        if ampls.shape[0] == 240:

            # z1s = ampls - Aref
            # z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
            # z1s = np.delete(z1s, ref_spot, axis=1)

            # z1s = ampls.copy()
            # z1s /= z1s[ref_spot]
            # z1s = np.delete(z1s, ref_spot, axis=1)

            # z1s /= z0_norm.copy()[None, :]

            Iall = ampls.copy() ** 2
            I0 = Iall[:, 0::4].copy()
            I1 = Iall[:, 1::4].copy()
            I2 = Iall[:, 2::4].copy()
            I3 = Iall[:, 3::4].copy()
            Xmeas = (I0 - I2) / scale_guess
            Ymeas = (I1 - I3) / scale_guess

            z1s = Xmeas + (1j*Ymeas)

            theories = np.dot(xs, slm_arr_Z.copy())
            # theories = np.delete(theories, ref_spot, axis=1)

            # theories /= z0_norm.copy()[None, :]

            all_z1s.append(z1s)
            all_theories.append(theories)

    all_z1s = np.array(all_z1s)
    all_z1s = all_z1s.reshape(all_z1s.shape[0]*240, 10)  # all_z1s.shape[0]*240, m-1
    all_theories = np.array(all_theories)
    all_theories = all_theories.reshape(all_theories.shape[0]*240, 10)  # all_z1s.shape[0]*240, m-1

    print(all_z1s.shape, all_theories.shape)

    np.save('./tools/temp_z1s.npy', all_z1s)
    np.save('./tools/temp_theories.npy', all_theories)

    def line(x, grad, c):
        return (grad * x) + c

    real_norm_params = np.array([curve_fit(line, np.real(all_theories[:, j]), np.real(all_z1s[:, j]))[0]
                                 for j in range(10)])
    imag_norm_params = np.array([curve_fit(line, np.imag(all_theories[:, j]), np.imag(all_z1s[:, j]))[0]
                                 for j in range(10)])

    Zreals = (np.real(all_z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
    Zimags = (np.imag(all_z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]

    all_z1s = Zreals + (1j * Zimags)

    error = np.real(all_z1s - all_theories).std()
    error_imag = np.imag(all_z1s - all_theories).std()
    print(colored('error : {:.3f}'.format(error), 'blue'))
    print(colored('error imag: {:.3f}'.format(error_imag), 'blue'))

    plt.ion()
    plt.show()

    fig3, [[axs0, axs1, axs2], [axs3, axs4, axs5]] = plt.subplots(2, 3, figsize=(8, 4))

    axs3.set_ylim(-10, 10)
    axs1.set_ylim(-10, 10)
    axs1.set_xlim(-10, 10)
    axs4.set_ylim(-10, 10)
    axs4.set_xlim(-10, 10)
    axs5.set_ylim(0, 100)
    axs5.set_xlim(0, 30)
    axs2.set_ylim(0, 5)
    axs2.set_xlim(0, 1500)

    axs1.plot([-10, 10], [-10, 10], c='black')
    eg_line = [axs1.plot(np.real(all_theories[:, j]), np.real(all_z1s[:, j]), linestyle='', marker='.', markersize=1)[0]
               for j in range(10)]  # m - 1

    axs4.plot([-10, 10], [-10, 10], c='black')
    eg_line_imag = [axs4.plot(np.imag(all_theories[:, j]), np.imag(all_z1s[:, j]), linestyle='', marker='.',
                              markersize=1)[0] for j in range(10)]  # m - 1

    th_line = axs3.plot(np.real(all_theories[0, :]), linestyle='', marker='o', c='b')[0]
    meas_line = axs3.plot(np.real(all_z1s[0, :]), linestyle='', marker='x', c='r')[0]

    th_line_imag = axs3.plot(np.imag(all_theories[0, :]), linestyle='', marker='o', c='g')[0]
    meas_line_imag = axs3.plot(np.imag(all_z1s[0, :]), linestyle='', marker='x', c='orange')[0]

    img = axs0.imshow(frames[0], aspect='auto')

    plt.draw()
    plt.pause(0.1)

    # plt.show()

    # input()

    # input('press anything to start...')

    ###########################
    # loop batches and epochs #
    ###########################

    num_batches = 50
    num_frames = 10
    batch_size = 240
    num_epochs = 20

    lim_arr = uppers1_ann.copy()

    # np.random.seed(100)
    # w1 = np.random.normal(0, 0.5, (n, m-1))
    # w1 = np.clip(w1, -lim_arr, lim_arr)

    np.random.seed(100)
    w1_x = np.random.normal(0, 0.5, (n, 10))
    w1_x = np.clip(w1_x, -lim_arr, lim_arr)
    np.random.seed(101)
    w1_y = np.random.normal(0, 0.5, (n, 10))
    w1_y = np.clip(w1_y, -lim_arr, lim_arr)

    # np.random.seed(200)
    # w2 = np.random.normal(0, 0.5, (m - 1, 10))

    # np.save('D:/MNIST/data/w1_0.npy', w1)
    # np.save('D:/MNIST/data/w2_0.npy', w2)

    np.save('D:/MNIST/data/w1_x_0.npy', w1_x)
    np.save('D:/MNIST/data/w1_y_0.npy', w1_y)

    # np.random.seed(321)
    # sys_noise_arr = np.random.normal(0, 0.2, w1.shape)
    # sys_noise_arr = np.random.normal(1., 0.5, w1.shape)

    # np.save('D:/MNIST/data/sys_noise_arr.npy', sys_noise_arr)

    # m_dw1 = np.zeros((n, m-1))
    # v_dw1 = np.zeros((n, m-1))

    # m_dw2 = np.zeros((m-1, 10))
    # v_dw2 = np.zeros((m-1, 10))

    m_dw_x_1 = np.zeros((n, 10))
    v_dw_x_1 = np.zeros((n, 10))
    m_dw_y_1 = np.zeros((n, 10))
    v_dw_y_1 = np.zeros((n, 10))

    beta1 = 0.9
    beta2 = 0.999
    # adam_params = (m_dw1, v_dw1, beta1, beta2)
    # adam_params = (m_dw1, v_dw1, m_dw2, v_dw2, beta1, beta2)
    adam_params = (m_dw_x_1, v_dw_x_1, m_dw_y_1, v_dw_y_1, beta1, beta2)

    # dnn = DNN_1d(*adam_params, x=trainX, y=trainY, w1_0=w1, batch_size=batch_size, num_batches=num_batches, lr=10e-3)

    # dnn = DNN(*adam_params, x=trainX, y=trainY, w1_0=w1, w2_0=w2, batch_size=batch_size, num_batches=num_batches,
    #           lr=20e-3, nonlinear=True)

    dnn = DNN_complex(*adam_params, x=trainX, y=trainY, w1_x_0=w1_x, w1_y_0=w1_y,
                      batch_size=batch_size, num_batches=num_batches, lr=10e-3, scaling=.6)

    loss = [5]
    errors = []
    errors_imag = []
    accs = []

    axs5.plot(accs, linestyle='-', marker='x', c='g')
    axs2.plot(loss, linestyle='-', marker='', c='r')
    # axs4.set_ylim(0, 0.5)
    # axs4.plot(errors, linestyle='-', marker='', c='b')
    # axs4.plot(errors_imag, linestyle='-', marker='', c='cyan')

    loop_clock = clock.Clock()
    loop_clock.tick()

    for epoch_num in range(num_epochs):

        epoch_start_time = time.time()

        w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())
        # CHECK REFERENCE IS ON/OFF
        update_slm(w1_z, lut=True, ref=False)
        time.sleep(1)

        epoch_rand_indxs = np.arange(60000)
        random.Random(epoch_num).shuffle(epoch_rand_indxs)
        batch_indxs_list = [epoch_rand_indxs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

        target_frames = cp.zeros((14, 1080, 1920, 4), dtype=cp.uint8)
        target_frames[..., -1] = 255

        # init loop
        fc = len(target_frames)-1
        cp_arr = target_frames[0]
        frame_count = 0

        for _ in range(5):
            capture.frames = []

            app.run(clock=dmd_clock, framerate=0, framecount=fc)

            time.sleep(0.1)
            loop_clock.tick()

        all_z1s = []
        all_theories = []

        for k in range(3):

            batch_indxs = np.random.randint(0, 60000, batch_size)
            # CHECK REFERENCE IS ON/OFF
            target_frames[2:-2, :, :, :-1] = make_dmd_batch(trainX_cp[batch_indxs, :], 0, ref_block_val, batch_size,
                                                            num_frames)

            xs = trainX[batch_indxs, :].copy()

            fc = target_frames.shape[0] - 1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []
            app.run(clock=dmd_clock, framerate=0, framecount=fc)
            time.sleep(0.1)

            frames = np.array(capture.frames.copy())

            ampls = find_spot_ampls(frames)

            print(ampls.shape)

            if ampls.shape[0] == 240:

                # z1s = ampls.copy()
                # # z1s /= z1s[ref_spot]
                # z1s = np.delete(z1s, ref_spot, axis=1)
                # z1s = (z1s - norm_params[:, 1].copy()) / norm_params[:, 0].copy()
                #
                # theories = np.dot(xs, dnn.w1.copy())

                Iall = ampls.copy() ** 2
                I0 = Iall[:, 0::4].copy()
                I1 = Iall[:, 1::4].copy()
                I2 = Iall[:, 2::4].copy()
                I3 = Iall[:, 3::4].copy()
                Xmeas = (I0 - I2) / scale_guess
                Ymeas = (I1 - I3) / scale_guess

                z1s = Xmeas + (1j * Ymeas)

                Zreals = (np.real(z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
                Zimags = (np.imag(z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]
                z1s = Zreals + (1j * Zimags)

                theories = np.dot(xs, w1_z.copy())

                all_z1s.append(z1s)
                all_theories.append(theories)

        all_z1s = np.array(all_z1s).reshape(3*240, 10)
        all_theories = np.array(all_theories).reshape(3*240, 10)

        print(all_z1s.shape)
        print(all_theories.shape)

        def line(x, grad, c):
            return (grad * x) + c
        #
        # norm_params_adjust = np.array([curve_fit(line, all_theories[..., j].flatten(), all_z1s[..., j].flatten())[0]
        #                                for j in range(m - 1)])
        # norm_params[:, 1] += norm_params[:, 0].copy() * norm_params_adjust[:, 1].copy()
        # norm_params[:, 0] *= norm_params_adjust[:, 0].copy()

        real_norm_params_adjust = np.array([curve_fit(line, np.real(all_theories[:, j]), np.real(all_z1s[:, j]))[0]
                                            for j in range(10)])
        imag_norm_params_adjust = np.array([curve_fit(line, np.imag(all_theories[:, j]), np.imag(all_z1s[:, j]))[0]
                                            for j in range(10)])

        real_norm_params[:, 1] += real_norm_params[:, 0].copy() * real_norm_params_adjust[:, 1].copy()
        real_norm_params[:, 0] *= real_norm_params_adjust[:, 0].copy()

        imag_norm_params[:, 1] += imag_norm_params[:, 0].copy() * imag_norm_params_adjust[:, 1].copy()
        imag_norm_params[:, 0] *= imag_norm_params_adjust[:, 0].copy()


        # w1_noisy = dnn.w1.copy()
        # w1_noisy += np.random.normal(0, 0.3, w1_noisy.shape)
        # w1_noisy += sys_noise_arr.copy()
        # w1_noisy *= sys_noise_arr.copy()
        # update_slm(w1_noisy, lut=True, ref=True)
        # time.sleep(1)

        ampl_noise = np.random.normal(0, 0.3, (n, m))

        w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())
        # CHECK REFERENCE IS ON/OFF
        update_slm(w1_z, lut=True, ref=False, noise_arr_A=None, noise_arr_phi=None)
        time.sleep(1)

        ##########################

        for batch_num in range(num_batches):

            t0 = time.time()

            print(batch_num)

            vecs = trainX_cp[batch_indxs_list[batch_num], :].copy()
            # CHECK REFERENCE IS ON/OFF
            target_frames[2:-2, ..., :-1] = make_dmd_batch(vecs, 0, ref_block_val, batch_size, num_frames)

            fc = target_frames.shape[0] - 1
            cp_arr = target_frames[0]
            frame_count = 0

            t1 = time.time()
            print(t1 - t0)
            #
            capture.frames = []
            app.run(clock=dmd_clock, framerate=0, framecount=fc)
            time.sleep(0.1)
            frames = np.array(capture.frames, dtype=np.uint8)

            # frames = np.load('./tools/frames_temp.npy').astype(np.uint8)

            ampls = find_spot_ampls(frames)

            np.save('D:/MNIST/raw_images/training/images/images_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), frames)
            np.save('D:/MNIST/data/training/ampls/ampls_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), ampls)

            t3 = time.time()
            print(t3 - t1)

            xs = trainX[batch_indxs_list[batch_num], :].copy()
            ys = trainY[batch_indxs_list[batch_num], :].copy()

            if ampls.shape[0] == batch_size:

                meas = ampls.copy().reshape((num_frames, batch_size//num_frames, m))
                diffs = np.abs(np.array([meas[k + 1, :, m//3] - meas[k, :, m//3]
                                         for k in range(num_frames-1)])).mean(axis=1)
                diffs /= diffs.max()
                repeats = (diffs < 0.5).sum() > 0

                if repeats:
                    print(colored('repeated frames, skipping', 'red'))

            else:
                print(colored('wrong num frames: {}'.format(ampls.shape[0]), 'red'))

            if ampls.shape[0] == batch_size and not repeats:

                # z1s = ampls.copy()
                # # z1s /= z1s[ref_spot]
                # z1s = np.delete(z1s, ref_spot, axis=1)
                # z1s = (z1s - norm_params[:, 1].copy()) / norm_params[:, 0].copy()
                #
                # theories = np.dot(xs, dnn.w1.copy())

                Iall = ampls.copy() ** 2
                I0 = Iall[:, 0::4].copy()
                I1 = Iall[:, 1::4].copy()
                I2 = Iall[:, 2::4].copy()
                I3 = Iall[:, 3::4].copy()
                Xmeas = (I0 - I2) / scale_guess
                Ymeas = (I1 - I3) / scale_guess

                z1s = Xmeas + (1j * Ymeas)

                Zreals = (np.real(z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
                Zimags = (np.imag(z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]
                z1s = Zreals + (1j * Zimags)

                theories = np.dot(xs, w1_z.copy())

                error = np.real(z1s - theories).std()
                error_imag = np.imag(z1s - theories).std()
                print(colored('error : {:.3f}, error imag: {:.3f}'.format(error, error_imag), 'blue'))
                errors.append(error)
                errors_imag.append(error_imag)

                t31 = time.time()
                print('calc zs: ', t31 - t3)

                dnn.feedforward(z1s)

                t32 = time.time()
                print('forward: ', t32 - t31)

                dnn.backprop(xs, ys)

                t33 = time.time()
                print('backward: ', t33 - t32)

                # dnn.w1 = np.clip(dnn.w1.copy(), -uppers1_ann, uppers1_ann)

                dnn.w1_x = np.clip(dnn.w1_x.copy(), -uppers1_ann, uppers1_ann)
                dnn.w1_y = np.clip(dnn.w1_y.copy(), -uppers1_ann, uppers1_ann)

                # w1_noisy = dnn.w1.copy()
                # w1_noisy += np.random.normal(0, 0.3, w1_noisy.shape)
                # w1_noisy += sys_noise_arr.copy()
                # w1_noisy *= sys_noise_arr.copy()
                # update_slm(w1_noisy, lut=True, ref=True)
                # time.sleep(0.7)

                ampl_noise = np.random.normal(0, 0.3, (n, m))

                w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())
                # CHECK REFERENCE IS ON/OFF
                update_slm(w1_z, lut=True, ref=False, noise_arr_A=None, noise_arr_phi=None)


                if dnn.loss < loss[-1]:
                    # best_w1 = dnn.w1.copy()
                    # best_w2 = dnn.w2.copy()
                    best_w1_x = dnn.w1_x.copy()
                    best_w1_y = dnn.w1_y.copy()

                loss.append(dnn.loss)
                print(colored('loss : {:.2f}'.format(dnn.loss), 'green'))
                np.save('D:/MNIST/data/loss.npy', np.array(loss))

                # new_adam_params = np.array([dnn.m_dw1, dnn.v_dw1, dnn.m_dw2, dnn.v_dw2, dnn.beta1, dnn.beta2])
                # new_adam_params = np.array([dnn.m_dw1, dnn.v_dw1, dnn.beta1, dnn.beta2])

                new_adam_params = np.array([dnn.m_dw_x_1, dnn.v_dw_x_1, dnn.m_dw_y_1, dnn.v_dw_y_1, dnn.beta1, dnn.beta2])

                np.save('D:/MNIST/data/adam_params.npy', new_adam_params)

            else:
                # z1s = np.full((batch_size, m-1), np.nan)
                # theories = np.full((batch_size, m-1), np.nan)

                z1s = np.full((batch_size, 10), np.nan+(1j*np.nan))
                theories = np.full((batch_size, 10), np.nan+(1j*np.nan))

                t33 = time.time()

            t4 = time.time()
            print('save: ', t4 - t33)

            np.save('D:/MNIST/data/training/measured/measured_arr_epoch_{}_batch_{}.npy'
                    .format(epoch_num, batch_num), z1s)
            np.save('D:/MNIST/data/training/theory/theory_arr_epoch_{}_batch_{}.npy'
                    .format(epoch_num, batch_num), theories)

            # np.save('D:/MNIST/data/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1))
            # np.save('D:/MNIST/data/w2/w2_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w2))

            np.save('D:/MNIST/data/w1/w1_x_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1_x))
            np.save('D:/MNIST/data/w1/w1_y_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1_y))

            for j in range(10): # m - 1
                eg_line[j].set_xdata(np.real(theories[:, j]))
                eg_line[j].set_ydata(np.real(z1s[:, j]))

                eg_line_imag[j].set_xdata(np.imag(theories[:, j]))
                eg_line_imag[j].set_ydata(np.imag(z1s[:, j]))

            th_line.set_ydata(np.real(theories[0, :]))
            meas_line.set_ydata(np.real(z1s[0, :]))

            th_line_imag.set_ydata(np.imag(theories[0, :]))
            meas_line_imag.set_ydata(np.imag(z1s[0, :]))

            img.set_array(frames[0])

            plt.draw()
            # plt.pause(0.001)
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.001)

            t5 = time.time()
            print(t5 - t4)

            dt = loop_clock.tick()
            print(colored(dt, 'yellow'))
            print()


        ##################
        ### VALIDATION ###
        ##################

        # dnn.w1 = best_w1.copy()
        # dnn.w2 = best_w2.copy()

        dnn.w1_x = best_w1_x.copy()
        dnn.w1_y = best_w1_y.copy()

        # np.save('D:/MNIST/data/best_w1.npy', best_w1)
        # np.save('D:/MNIST/data/best_w2.npy', best_w2)

        np.save('D:/MNIST/data/best_w1_x.npy', best_w1_x)
        np.save('D:/MNIST/data/best_w1_y.npy', best_w1_y)


        # w1_noisy = dnn.w1.copy()
        # w1_noisy += np.random.normal(0, 0.3, w1_noisy.shape)
        # w1_noisy += sys_noise_arr.copy()
        # w1_noisy *= sys_noise_arr.copy()
        # update_slm(w1_noisy, lut=True, ref=True)
        # time.sleep(0.7)

        ampl_noise = np.random.normal(0, 0.3, (n, m))

        w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())
        # CHECK REFERENCE IS ON/OFF
        update_slm(w1_z, lut=True, ref=False, noise_arr_A=None, noise_arr_phi=None)
        time.sleep(1)

        # val_z1s = np.full((4800, m-1), np.nan)
        val_z1s = np.full((4800, 10), np.nan+(1j*np.nan))

        for val_batch_num in range(20):

            print(val_batch_num)

            vecs = valX_cp[val_batch_num * 240:(val_batch_num + 1) * 240, :].copy()

            # CHECK REFERENCE IS ON/OFF
            target_frames[2:-2, ..., :-1] = make_dmd_batch(vecs, 0, ref_block_val, batch_size, num_frames)

            fc = target_frames.shape[0] - 1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []

            app.run(clock=dmd_clock, framerate=0, framecount=fc)

            time.sleep(0.1)

            frames = np.array(capture.frames.copy(), dtype=np.uint8)
            ampls = find_spot_ampls(frames)

            np.save('D:/MNIST/raw_images/validation/images/images_epoch_{}_batch_{}.npy'
                    .format(epoch_num, val_batch_num), frames)
            np.save('D:/MNIST/data/validation/ampls/ampls_epoch_{}_batch_{}.npy'
                    .format(epoch_num, val_batch_num), ampls)

            xs = valX[val_batch_num * 240:(val_batch_num + 1) * 240, :].copy()

            if ampls.shape[0] == batch_size:

                meas = ampls.copy().reshape((num_frames, batch_size // num_frames, m))
                diffs = np.abs(np.array([meas[k + 1, :, m // 3] - meas[k, :, m // 3]
                                         for k in range(num_frames - 1)])).mean(axis=1)
                diffs /= diffs.max()
                repeats = (diffs < 0.5).sum() > 0

                if repeats:
                    print(colored('repeated frames, skipping', 'red'))

            else:
                print(colored('wrong num frames: {}'.format(ampls.shape[0]), 'red'))

            if ampls.shape[0] == batch_size and not repeats:

                # z1s = np.delete(ampls, ref_spot, axis=1)
                # z1s = (z1s - norm_params[:, 1].copy()) / norm_params[:, 0].copy()

                Iall = ampls.copy() ** 2
                I0 = Iall[:, 0::4].copy()
                I1 = Iall[:, 1::4].copy()
                I2 = Iall[:, 2::4].copy()
                I3 = Iall[:, 3::4].copy()
                Xmeas = (I0 - I2) / scale_guess
                Ymeas = (I1 - I3) / scale_guess

                z1s = Xmeas + (1j * Ymeas)

                Zreals = (np.real(z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
                Zimags = (np.imag(z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]
                z1s = Zreals + (1j * Zimags)

                val_z1s[val_batch_num * 240:(val_batch_num + 1) * 240, :] = z1s.copy()

                # theories = np.dot(xs, dnn.w1.copy())
                theories = np.dot(xs, w1_z.copy())

            else:
                z1s = np.full((batch_size, 10), np.nan+(1j*np.nan))  # m - 1
                theories = np.full((batch_size, 10), np.nan+(1j*np.nan))  # m - 1

            np.save('D:/MNIST/data/validation/measured/measured_arr_epoch_{}_batch_{}.npy'
                    .format(epoch_num, val_batch_num), z1s)
            np.save('D:/MNIST/data/validation/theory/theory_arr_epoch_{}_batch_{}.npy'
                    .format(epoch_num, val_batch_num), theories)

        mask = ~np.isnan(np.real(val_z1s)[:, 0])
        val_z1s = val_z1s[mask]
        # xs = valX[:4800].copy()[mask]
        ys = valY[:4800].copy()[mask]

        dnn.feedforward(val_z1s)

        # pred = dnn.a1.argmax(axis=1)
        pred = dnn.a2.argmax(axis=1)
        label = ys.argmax(axis=1)

        acc = accuracy(pred, label)
        accs.append(acc)

        np.save('D:/MNIST/data/accuracy.npy', np.array(accs))

        axs5.plot(accs, linestyle='-', marker='x', c='g')
        axs2.plot(loss, linestyle='-', marker='', c='r')
        # axs4.plot(errors, linestyle='-', marker='', c='b')
        plt.draw()
        plt.pause(0.001)

        epoch_time = time.time() - epoch_start_time

        print('\n######################################################################')
        print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                      .format(epoch_num, epoch_time, accs[-1], loss[-1]), 'green'))
        print('######################################################################\n')


    print()
    camera.Close()
    # imageWindow.Close()
    context.pop()


