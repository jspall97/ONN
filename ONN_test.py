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
from pylon import run_camera, view_camera
from ANN import DNN
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])
l = int(dims[2])


inputs = loadmat('./tools/MNIST digit - subsampled - 121.mat')

trainX_raw = inputs['trainX']
trainY_raw = inputs['trainY']
testX_raw = inputs['testX']
testY_raw = inputs['testY']

num_train = 60000
num_test = 10000

trainY = np.zeros((num_train, 10))
testY = np.zeros((num_test, 10))

for i in range(num_train):
    trainY[i, trainY_raw[0, i]] = 1

for i in range(num_test):
    testY[i, testY_raw[0, i]] = 1

trainX = np.empty((num_train, 121))
for i in range(num_train):
    trainX_k = trainX_raw[i, :] - trainX_raw[i, :].min()
    trainX_k = trainX_k / trainX_k.max()
    trainX[i, :] = 1 - trainX_k

testX = np.empty((num_test, 121))
for i in range(num_test):
    testX_k = testX_raw[i, :] - testX_raw[i, :].min()
    testX_k = testX_k / testX_k.max()
    testX[i, :] = 1 - testX_k

random.Random(0).shuffle(trainX)
random.Random(0).shuffle(trainY)
random.Random(0).shuffle(testX)
random.Random(0).shuffle(testY)

valX = testX[:5000, :].copy()
testX = testX[5000:, :].copy()

valY = testY[:5000, :].copy()
testY = testY[5000:, :].copy()



def keyboardinterrupthandler(signal, frame):
    context.pop()
    print('keyboard interupt - closed')
    exit(0)


signal.signal(signal.SIGINT, keyboardinterrupthandler)


def run_frames(frames, fr=0, fc=None, pad=True):
    global cp_arr, target_frames, frame_count, batch_num

    def pad_frames(frs, lead, trail):
        global null_frame
        frs[0:0] = [null_frame for _ in range(lead)]
        frs.extend([null_frame for _ in range(trail)])
        return frs

    if pad:
        target_frames = pad_frames(frames, 3, 2)
    else:
        target_frames = frames

    if fc is None:
        fc = len(frames) - 2

    cp_arr = target_frames[0]
    frame_count = 0

    app.run(framerate=fr, framecount=fc)

    print('dmd finished batch {}'.format(batch_num))


actual_uppers_arr_1024 = np.load('./tools/actual_uppers_arr_1024.npy')

uppers1_nm = actual_uppers_arr_1024[..., -1].copy()

actual_uppers_arr_1024_T = np.transpose(actual_uppers_arr_1024, (2, 0, 1))
gpu_actual_uppers_arr_1024 = cp.asarray(actual_uppers_arr_1024_T)


def ampl_lut_nm(arr_in):
    gpu_arr = cp.asarray(arr_in)
    map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_1024 - cp.abs(gpu_arr)), axis=0)
    arr_out = cp.linspace(0, 1, 1024)[map_indx]

    return arr_out


def update_slm(arr, lut=False, stag=True):

    if lut:
        arr_A = ampl_lut_nm(arr)
        arr_phi = cp.asarray(np.angle(arr))
        img = make_slm_rgb(arr_A, arr_phi, stagger=stag).get()
    else:
        arr_A = cp.asarray(arr)
        img = make_slm_rgb(arr_A, stagger=stag).get()

    slm.updateArray(img)
    time.sleep(0.7)


def wait_for_sig(conn):
    while not conn.poll():
        pass
    conn.recv()


def dmd_one_frame(arr):
    img = make_dmd_image(arr)
    frame = make_dmd_rgb([img for _ in range(24)])
    return [frame]


def init_cam_dmd_scan():
    global conn1, conn2, batch_num, null_frame, null_frames

    conn1, conn2 = Pipe()
    cam_process = Process(target=run_camera, args=[conn2, 'train'])
    cam_process.daemon = 1
    cam_process.start()
    time.sleep(5)

    batch_num = 0
    null_frame = dmd_one_frame(np.ones((n, m)))[0]
    null_frames = [null_frame for _ in range(200)]

    print('INIT CAMERA NULL FRAME')
    conn1.send((0, 500))
    run_frames(frames=null_frames)  # init frames, "batch" 0
    wait_for_sig(conn1)


def process_frames(ampls_arr, expected_num, noise_level=0.5, sim_level=5):

    diffs = np.diff(ampls_arr[:, m // 2])
    large_diffs = np.where(np.abs(diffs) > noise_level)[0]
    start_indx = large_diffs[0] + 1
    end_indx = large_diffs[-1] + 1

    print(start_indx, end_indx, end_indx - start_indx)

    ampls_arr_cut = ampls_arr[start_indx:end_indx, :].copy()

    num_f = ampls_arr_cut.shape[0] // 24

    assert num_f == expected_num

    ampls_arr_split = [ampls_arr_cut[i * 24:(i + 1) * 24, :] for i in range(num_f)]

    frame_similarity = np.array([np.linalg.norm(ampls_arr_split[i] - ampls_arr_split[i + 1])
                                 for i in range(num_f - 1)])

    print(frame_similarity)

    repeats = np.where(frame_similarity < sim_level)[0]

    # if len(repeats > 0):
    #     if num_frames == batch_size:
    #         print('deleting frames', repeats)
    #         for indx in repeats:
    #             ampls_arr_cut[(indx) * 24:(indx + 1) * 24, :] = np.NaN
    #
    #     else:
    #         repeats_ext = repeats #np.unique(np.hstack((repeats, repeats - 1)))
    #         print('deleting frames', repeats_ext)
    #         for indx in repeats_ext:
    #             ampls_arr_cut[indx * 24:(indx + 1) * 24, :] = np.NaN

    if len(repeats > 0):
        print('deleting frames', repeats)
        for indx in repeats:
            ampls_arr_cut[(indx) * 24:(indx + 1) * 24, :] = np.NaN


    return ampls_arr_cut, diffs, repeats



if __name__ == '__main__':

    ################
    # SLM display #
    ################

    slm = SLMdisplay(0)
    slm_img = np.full((n, m), 1.)
    update_slm(slm_img, lut=False)

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

    batch_num = 0
    #
    # null_frame = dmd_one_frame(np.ones((n, m)))[0]
    # null_frames = [null_frame for _ in range(20)]
    # run_frames(frames=null_frames, fc=100000)  # init frames, "batch" 0

    # slm_arr = np.full((n, m), 1)
    # slm_arr[:n//2, :m//2] = 0
    # update_slm(slm_arr, lut=False)
    #
    # eg_frame = [cp.array(np.load('./MNIST/trainX/frames/rgb24_batch_{}.npy'.format(0))[0, ...])]
    # run_frames(frames=eg_frame, pad=False, fc=10000)


    #####################
    # find norm values #
    #####################

    ampl_norm_val = 0.5

    dmd_frame = dmd_one_frame(np.ones((n, m)))
    null_frames = [dmd_frame[0] for _ in range(100)]
    run_frames(frames=null_frames, pad=False)

    w1 = np.full((n, m), ampl_norm_val)
    update_slm(w1, lut=False)

    # view_camera()

    ######################

    k = int(np.argmin(np.abs(np.linspace(0, 1, 1024) - ampl_norm_val)))

    theory = actual_uppers_arr_1024[..., k].sum(axis=0)
    theory_norm_val = theory.max()
    indx = np.argmax(theory)

    init_cam_dmd_scan()
    print()

    dmd_frame = dmd_one_frame(np.ones((n, m)))
    run_frames(frames=dmd_frame, pad=False)

    print('find norm values')
    conn1.send((2, 200))
    wait_for_sig(conn1)

    loaded = np.load('./MNIST/pylon_captures/ampls/batch_{}.npy'.format(0))
    meas = loaded.mean(axis=0)

    print(meas)

    meas_norm_val = meas[indx]

    print(theory_norm_val, meas_norm_val)

    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 4))
    axs2.set_ylim(0, 1.2)
    axs2.plot(theory/theory_norm_val, linestyle='', marker='o', c='b')
    axs2.plot(meas/meas_norm_val, linestyle='', marker='x', c='r')
    plt.draw()

    plt.show()


    print('############')
    print()

    #####################
    # loop over batches #
    #####################

    num_batches = 10
    num_frames = 10

    slm_arr = uppers1_nm.copy() * 0.6 # slm_rm_case_10(0)
    update_slm(slm_arr, lut=True)

    null_frames = [null_frame for _ in range(200)]
    conn1.send((2, 1000))  # 0 used for init, 1 used to terminate
    run_frames(frames=null_frames)
    wait_for_sig(conn1)

    for batch_num in range(num_batches):

        batch_frames = [cp.array(np.load('./MNIST/test_frames/frames/frames.npy')[i, ...])
                        for i in range(num_frames)]

        # half_frame = dmd_one_frame(np.full((n, m), 0.5))[0]
        # batch_frames = [half_frame for _ in range(num_frames)]

        conn1.send((batch_num+2, 450))  # 0 used for init, 1 used to terminate
        run_frames(frames=batch_frames)
        wait_for_sig(conn1)

        ampls = np.load('./MNIST/pylon_captures/ampls/batch_{}.npy'.format(batch_num))

        try:
            measureds, diffs, repeats = process_frames(ampls, num_frames, sim_level=0)
        except AssertionError:
            print('process error, wrong num frames')
            continue
        measureds *= ampl_norm_val*theory_norm_val/meas_norm_val

        dmd_targets = [np.load('./MNIST/test_frames/targets/target_arrs.npy')[i, ...]
                       for i in range(num_frames*24)]
        #
        # dmd_tar = (np.full((n, m), 0.5)*15).astype(int)/15
        # dmd_targets = [dmd_tar for i in range(num_frames*24)]

        theories = []
        for i in range(len(dmd_targets)):
            target = (slm_arr.copy() * dmd_targets[i]).sum(axis=0)
            theory = target * ampl_norm_val
            theories.append(theory)

        theories = np.array(theories)

        np.save('./MNIST/test_frames/measured/measured_arr_batch{}.npy'.format(batch_num), measureds)
        np.save('./MNIST/test_frames/theory/theory_arr_batch{}.npy'.format(batch_num), theories)

        # slm_arr = uppers1_nm.copy() * slm_rm_case_10(batch_num+1)
        # update_slm(slm_arr, lut=True)

        time.sleep(0.5)

        print()

    conn1.send(1)  # stop camera

    context.pop()
    print('closed dmd')
