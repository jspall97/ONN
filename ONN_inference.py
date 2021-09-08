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

ref_spot = m//2

ref_block_val = 0.
batch_size = 240
num_batches = 5
num_frames = 10

dmd_block_w = update_params(ref_block_val, batch_size, num_frames, is_complex=True)

# ref_block_val = None
# batch_size = 240
# num_batches = 5
# num_frames = 10
#
# dmd_block_w = update_params(ref_block_val, batch_size, num_frames, is_complex=False)

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

complex_output_ratios = np.load('./tools/complex_output_ratios.npy')


def update_slm(arr, lut=False, ref=False):
    global ampl_norm_val

    if arr.shape[1] == m - 1:
        arr = np.insert(arr, ref_spot, np.zeros(n), 1)

    if arr.shape[1] == 10:
        arr = np.repeat(arr.copy(), 4, axis=1) * complex_output_ratios.copy()[None, :]

    if lut:
        # gpu_arr = cp.asarray(arr.copy())
        # map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_256 - gpu_arr), axis=0)
        # arr_A = cp.linspace(-1., 1., 256)[map_indx]

        # gpu_arr = cp.abs(cp.asarray(arr.copy()))
        # map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_256 - gpu_arr), axis=0)
        # arr_A = cp.linspace(-1., 1., 256)[map_indx]

        gpu_arr = cp.abs(cp.asarray(arr.copy()))
        map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_128_flat - gpu_arr), axis=0)
        arr_A = cp.linspace(0, 1, 128)[map_indx]

    else:
        arr_A = cp.asarray(arr.copy())

    arr_out = arr_A * cp.exp(1j * cp.angle(cp.array(arr.copy())))

    if ref:
        arr_out[:, ref_spot] = ampl_norm_val

    arr_out = np.flip(arr_out.get(), axis=1)
    img = make_slm_rgb(arr_out, ref_block_val)
    slm.updateArray(img)
    # time.sleep(0.7)


def dmd_one_frame(arr, ref):
    img = make_dmd_image(arr, ref=ref, ref_block_val=ref_block_val)
    return [img]


x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
y_centers = np.load('./tools/y_centers_list.npy')


def find_spot_ampls(arrs):

    # arrs = np.array([recombine(arr.T) for arr in arrs_in])

    mask = arrs < 3
    arrs -= 2
    arrs[mask] = 0

    def spot_s(i):
        return np.s_[:, y_centers[i] - 1:y_centers[i] + 2, x_edge_indxs[2 * i]:x_edge_indxs[2 * i + 1]]

    spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(m + 1)])

    spot_ampls = cp.sqrt(spot_powers)

    spot_ampls = np.flip(spot_ampls, axis=0)

    ratio = spot_ampls[ref_spot, :] / spot_ampls[ref_spot + 1, :]

    spot_ampls[ref_spot + 1:, :] *= ratio[None, :]

    spot_ampls = np.delete(spot_ampls.get(), ref_spot, 0)

    return spot_ampls.T


# actual_uppers_arr_256 = np.load("C:/Users/spall/PycharmProjects/ONN/tools/actual_uppers_arr_256.npy")

# actual_uppers_arr_256[:, :, ref_spot] = actual_uppers_arr_256[:, :, ref_spot + 1]

# uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
# uppers1_ann = np.delete(uppers1_nm, ref_spot, 1)

# k = np.abs(np.linspace(-1, 1, 256) - 0.1).argmin()
# z0 = actual_uppers_arr_256[k, ...].sum(axis=0)

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


    ######################################

    # Aref = np.load('./tools/Aref.npy')
    ampl_norm_val = 0.05

    scale_guess = 0.21

    # ref_block_val = 0.
    # dmd_block_w = update_params(ref_block_val, batch_size, num_frames)

    batch_size = 240
    num_frames = 10

    # w1 = np.load('D:/MNIST/data/best_w1_offline.npy')
    # w2 = np.load('D:/MNIST/data/best_w2_offline.npy')

    # w1_x = np.load('D:/MNIST/data/best_w1_x.npy')
    # w1_y = np.load('D:/MNIST/data/best_w1_y.npy')

    w1_x = np.load('D:/MNIST/data/best_w1_x_offline.npy')
    w1_y = np.load('D:/MNIST/data/best_w1_y_offline.npy')

    w1_z = w1_x.copy() + (1j * w1_y.copy())

    update_slm(w1_z, lut=True, ref=False)
    time.sleep(0.7)

    accs = []

    for rep in range(1):

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

            print(k)

            batch_indxs = np.random.randint(0, 5000, batch_size)

            target_frames[4:-2, :, :, :-1] = make_dmd_batch(testX_cp[batch_indxs, :], 0, ref_block_val, batch_size,
                                                            num_frames)

            xs = testX[batch_indxs, :].copy()

            fc = target_frames.shape[0] - 1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []
            app.run(clock=dmd_clock, framerate=0, framecount=fc)
            time.sleep(0.1)

            frames = np.array(capture.frames.copy())

            print(frames.shape)

            ampls = find_spot_ampls(frames)


            if ampls.shape[0] == 240:

                print(ampls.shape)

                # z1s = ampls - Aref
                # z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
                # z1s = np.delete(z1s, ref_spot, axis=1)
                #
                # theories = np.dot(xs, w1)

                # Iall = ampls.copy() ** 2
                # I0 = Iall[:, 0::4].copy()
                # I1 = Iall[:, 1::4].copy()
                # I2 = Iall[:, 2::4].copy()
                # I3 = Iall[:, 3::4].copy()
                # Xmeas = (I0 - I2) / scale_guess
                # Ymeas = (I1 - I3) / scale_guess
                #
                # z1s = Xmeas + (1j * Ymeas)
                #
                # theories = np.dot(xs, w1_z.copy())
                #
                # all_z1s.append(z1s)
                # all_theories.append(theories)

                Imeas = ampls.copy() ** 2
                Imeas = Imeas.reshape(240, 10, 4).mean(axis=-1)
                Imeas *= scale_guess
                z1s = Imeas.copy()

                theories = np.dot(xs, w1_z.copy())
                theories = np.abs(theories) ** 2

                all_z1s.append(z1s)
                all_theories.append(theories)

        all_theories = np.array(all_theories)
        all_z1s = np.array(all_z1s)

        print(all_z1s.shape)

        all_z1s = all_z1s.reshape(all_z1s.shape[0] * 240, 10)  # m - 1
        all_theories = all_theories.reshape(all_theories.shape[0] * 240, 10)  # m - 1

        print(all_z1s.shape, all_theories.shape)

        np.save('./tools/temp_z1s.npy', all_z1s)
        np.save('./tools/temp_theories.npy', all_theories)

        def line(x, grad, c):
            return (grad * x) + c
        #
        norm_params = np.array([curve_fit(line, all_theories[:, j], all_z1s[:, j])[0]
                                for j in range(10)]) # m - 1

        all_z1s -= norm_params[:, 1]
        all_z1s /= norm_params[:, 0]

        # real_norm_params = np.array([curve_fit(line, np.real(all_theories[:, j]), np.real(all_z1s[:, j]))[0]
        #                              for j in range(10)])
        # imag_norm_params = np.array([curve_fit(line, np.imag(all_theories[:, j]), np.imag(all_z1s[:, j]))[0]
        #                              for j in range(10)])
        #
        # Zreals = (np.real(all_z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
        # Zimags = (np.imag(all_z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]
        #
        # all_z1s = Zreals + (1j * Zimags)

        error = np.real(all_z1s - all_theories).std()
        # error_imag = np.imag(all_z1s - all_theories).std()
        print(colored('error : {:.3f}'.format(error), 'blue'))
        # print(colored('error imag: {:.3f}'.format(error_imag), 'blue'))

        print(colored('signal: {:.3f}'.format(all_theories.std()), 'blue'))

        # fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
        # axs3.set_ylim(-10, 10)
        # axs3.plot([-10, 10], [-10, 10], c='black')
        # for j in range(10):  # m - 1
        #     axs3.plot(all_theories[:, j], all_z1s[:, j], linestyle='', marker='.', markersize=1)
        # plt.draw()
        #
        # fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
        # axs4.set_ylim(-10, 10)
        # axs4.plot(all_theories[0, :], linestyle='', marker='o', c='b')
        # axs4.plot(all_z1s[0, :], linestyle='', marker='x', c='r')
        # plt.draw()

        lim = 80
        fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
        axs3.set_xlim(0, lim)
        axs3.set_ylim(0, lim)
        axs3.plot([0, lim], [0, lim], c='black')
        for j in range(10):  # m - 1
            axs3.plot(all_theories[:, j], all_z1s[:, j], linestyle='', marker='.', markersize=1)
        plt.draw()

        fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
        axs4.set_ylim(0, lim)
        axs4.plot(all_theories[0, :], linestyle='', marker='o', c='b')
        axs4.plot(all_z1s[0, :], linestyle='', marker='x', c='r')
        plt.draw()

        plt.show()

        # breakpoint()

        ###########
        # TESTING #
        ###########

        start_time = time.time()

        # np.random.seed(321)
        # sys_noise_arr = np.random.normal(0, 0.2, w1.shape)
        # sys_noise_arr = np.random.normal(1., 0.5, w1.shape)

        # w1_noisy = w1.copy()
        # w1_noisy += sys_noise_arr.copy()
        # w1_noisy += np.random.normal(0, 0.3, w1_noisy.shape)
        # w1_noisy *= sys_noise_arr.copy()
        # update_slm(w1_noisy, lut=True, ref=True)
        # time.sleep(0.7)

        #
        test_z1s = np.full((5000, 10), np.nan+(1j*np.nan))  # m - 1
        test_theories = np.full((5000, 10), np.nan+(1j*np.nan))

        # test_z1s = np.full((5000, 10), np.nan)
        # test_theories = np.full((5000, 10), np.nan)

        for test_batch_num in range(20):

            print(test_batch_num)
            vecs = testX_cp[test_batch_num * 240:(test_batch_num + 1) * 240, :].copy()

            print(vecs.shape)

            target_frames[4:-2, ..., :-1] = make_dmd_batch(vecs, 0, ref_block_val, batch_size, num_frames)

            fc = target_frames.shape[0] - 1
            cp_arr = target_frames[0]
            frame_count = 0

            capture.frames = []

            app.run(clock=dmd_clock, framerate=0, framecount=fc)

            time.sleep(0.1)

            frames = np.array(capture.frames.copy(), dtype=np.uint8)
            ampls = find_spot_ampls(frames)

            np.save('D:/MNIST/raw_images/testing/images/images_batch_{}.npy'
                    .format(test_batch_num), frames)
            np.save('D:/MNIST/data/testing/ampls/ampls_rep_{}_batch_{}.npy'
                    .format(rep, test_batch_num), ampls)

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

                # z1s = ampls - Aref
                # z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
                # z1s = np.delete(z1s, ref_spot, axis=1)
                #
                # z1s -= norm_params[:, 1]
                # z1s /= norm_params[:, 0]

                # Iall = ampls.copy() ** 2
                # I0 = Iall[:, 0::4].copy()
                # I1 = Iall[:, 1::4].copy()
                # I2 = Iall[:, 2::4].copy()
                # I3 = Iall[:, 3::4].copy()
                # Xmeas = (I0 - I2) / scale_guess
                # Ymeas = (I1 - I3) / scale_guess
                #
                # z1s = Xmeas + (1j * Ymeas)
                #
                # Zreals = (np.real(z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
                # Zimags = (np.imag(z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]
                # z1s = Zreals + (1j * Zimags)
                #
                # test_z1s[test_batch_num * 240:(test_batch_num + 1) * 240, :] = z1s.copy()
                #
                # theories = np.dot(xs, w1_z.copy())
                # test_theories[test_batch_num * 240:(test_batch_num + 1) * 240, :] = theories.copy()

                Imeas = ampls.copy() ** 2
                Imeas = Imeas.reshape(240, 10, 4).mean(axis=-1)
                Imeas *= scale_guess
                z1s = Imeas.copy()

                z1s -= norm_params[:, 1]
                z1s /= norm_params[:, 0]

                test_z1s[test_batch_num * 240:(test_batch_num + 1) * 240, :] = z1s.copy()

                theories = np.dot(xs, w1_z.copy())
                theories = np.abs(theories) ** 2

                test_theories[test_batch_num * 240:(test_batch_num + 1) * 240, :] = theories.copy()

            else:
                z1s = np.full((batch_size, 10), np.nan+(1j*np.nan))  # m - 1
                theories = np.full((batch_size, 10), np.nan+(1j*np.nan))  # m - 1

            np.save('D:/MNIST/data/testing/measured/measured_arr_rep_{}_batch_{}.npy'
                    .format(rep, test_batch_num), z1s)
            np.save('D:/MNIST/data/testing/theory/theory_arr_rep_{}_batch_{}.npy'
                    .format(rep, test_batch_num), theories)

        test_batch_num = 20

        print(test_batch_num)

        vecs = testX_cp[4800 - 40:, :].copy()
        vecs[:40, :] = 0.

        target_frames[4:-2, ..., :-1] = make_dmd_batch(vecs, 0, ref_block_val, batch_size, num_frames)

        fc = target_frames.shape[0] - 1
        cp_arr = target_frames[0]
        frame_count = 0

        capture.frames = []
        app.run(clock=dmd_clock, framerate=0, framecount=fc)
        time.sleep(0.1)

        frames = np.array(capture.frames.copy(), dtype=np.uint8)
        ampls = find_spot_ampls(frames)

        if ampls.shape[0] == 240:
            ampls = ampls[40:, :]

        np.save('D:/MNIST/raw_images/testing/images/images_batch_{}.npy'
                .format(test_batch_num), frames)
        np.save('D:/MNIST/data/testing/ampls/ampls_rep_{}_batch_{}.npy'
                .format(rep, test_batch_num), ampls)

        xs = testX[4800:, :].copy()

        if ampls.shape[0] == 200:

            meas = np.zeros((240, m))
            meas[:200, :] = ampls.copy()
            meas = meas.reshape((num_frames, 240 // num_frames, m))
            diffs = np.abs(np.array([meas[k + 1, :, m // 3] - meas[k, :, m // 3]
                                     for k in range(num_frames - 1)])).mean(axis=1)
            diffs /= diffs.max()
            repeats = (diffs < 0.25).sum() > 0

            if repeats:
                print(colored('repeated frames, skipping', 'red'))

        else:
            print(colored('wrong num frames: {}'.format(ampls.shape[0]), 'red'))

        if ampls.shape[0] == 200 and not repeats:

            # z1s = ampls - Aref
            # z1s = z1s * z0[ref_spot] / z1s[:, ref_spot][:, None]
            # z1s = np.delete(z1s, ref_spot, axis=1)
            #
            # z1s -= norm_params[:, 1]
            # z1s /= norm_params[:, 0]

            # Iall = ampls.copy() ** 2
            # I0 = Iall[:, 0::4].copy()
            # I1 = Iall[:, 1::4].copy()
            # I2 = Iall[:, 2::4].copy()
            # I3 = Iall[:, 3::4].copy()
            # Xmeas = (I0 - I2) / scale_guess
            # Ymeas = (I1 - I3) / scale_guess
            #
            # z1s = Xmeas + (1j * Ymeas)
            #
            # Zreals = (np.real(z1s).copy() - real_norm_params[:, 1]) / real_norm_params[:, 0]
            # Zimags = (np.imag(z1s).copy() - imag_norm_params[:, 1]) / imag_norm_params[:, 0]
            # z1s = Zreals + (1j * Zimags)
            #
            # test_z1s[4800:, :] = z1s.copy()
            #
            # theories = np.dot(xs, w1_z.copy())
            #
            # test_theories[4800:, :] = theories.copy()

            Imeas = ampls.copy() ** 2
            Imeas = Imeas.reshape(200, 10, 4).mean(axis=-1)
            Imeas *= scale_guess
            z1s = Imeas.copy()

            z1s -= norm_params[:, 1]
            z1s /= norm_params[:, 0]

            test_z1s[4800:, :] = z1s.copy()

            theories = np.dot(xs, w1_z.copy())
            theories = np.abs(theories) ** 2

            test_theories[4800:, :] = theories.copy()

        else:
            z1s = np.full((200, 10), np.nan+(1j*np.nan))  # m - 1
            theories = np.full((200, 10), np.nan+(1j*np.nan))  # m - 1

        np.save('D:/MNIST/data/testing/measured/measured_arr_rep_{}_batch_{}.npy'
                .format(rep, test_batch_num), z1s)
        np.save('D:/MNIST/data/testing/theory/theory_arr_rep_{}_batch_{}.npy'
                .format(rep, test_batch_num), theories)

        mask = ~np.isnan(np.real(test_z1s[:, 0]))
        test_z1s = test_z1s[mask]
        test_theories = test_theories[mask]

        # fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
        # axs3.set_ylim(-10, 10)
        # axs3.plot([-10, 10], [-10, 10], c='black')
        # for j in range(10):  # m - 1
        #     axs3.plot(test_theories[:, j], test_z1s[:, j], linestyle='', marker='.', markersize=1)
        # plt.draw()
        #
        # fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
        # axs4.set_ylim(-10, 10)
        # axs4.plot(test_theories[0, :], linestyle='', marker='o', c='b')
        # axs4.plot(test_z1s[0, :], linestyle='', marker='x', c='r')
        # plt.draw()
        #
        # plt.show()

        xs = testX.copy()[mask]
        ys = testY.copy()[mask]

        # # 1 layer
        # a1s = softmax(test_z1s*2)
        # pred = a1s.argmax(axis=1)

        # def relu(x):
        #     return np.maximum(0, x)
        #
        # # 2 layer
        # a1s = relu(test_z1s)
        # z2s = np.dot(a1s, w2)
        # a2s = softmax(z2s)
        # pred = a2s.argmax(axis=1)

        # complex
        scaling = 0.6
        #
        # z1_x = np.real(test_z1s.copy())
        # z1_y = np.imag(test_z1s.copy())
        #
        # a1_x = z1_x ** 2
        # a1_y = z1_y ** 2

        # z2 = a1_x + a1_y

        z2 = test_z1s.copy()
        # z2 = test_theories.copy()

        a2 = softmax(z2 * scaling)

        pred = a2.argmax(axis=1)
        label = ys.argmax(axis=1)

        acc = accuracy(pred, label)

        accs.append(acc)

        print('\n######################################################################')
        print(colored('time : {:.2f}, accuracy : {:.2f}'.format(time.time() - start_time, acc), 'green'))
        print('######################################################################\n')

        print()

    print(np.mean(np.array(accs)))

    camera.Close()
    # imageWindow.Close()
    context.pop()


