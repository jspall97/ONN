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

actual_uppers_arr_1024 = np.load('./tools/actual_uppers_arr_1024.npy')

uppers1_nm = actual_uppers_arr_1024[..., -1].copy()

actual_uppers_arr_1024_T = np.transpose(actual_uppers_arr_1024, (2, 0, 1))
gpu_actual_uppers_arr_1024 = cp.asarray(actual_uppers_arr_1024_T)


def ampl_lut_nm(arr_in):
    gpu_arr = cp.asarray(arr_in)
    map_indx = cp.argmin(cp.abs(gpu_actual_uppers_arr_1024 - cp.abs(gpu_arr)), axis=0)
    arr_out = cp.linspace(0, 1, 1024)[map_indx]

    return arr_out


def wait_for_sig(conn):
    while not conn.poll():
        pass
    conn.recv()


def dmd_one_frame(arr):
    img = make_dmd_image(arr)
    frame = make_dmd_rgb([img for _ in range(24)])
    return [frame]


def process_frames(ampls_arr, expected_num, noise_level=0.5, sim_level=5):

    diffs = np.diff(ampls_arr[:, m // 2])
    large_diffs = np.where(np.abs(diffs) > noise_level)[0]
    start_indx = large_diffs[0] + 1
    end_indx = large_diffs[-1] + 1

    print(start_indx, end_indx, end_indx - start_indx)

    ampls_arr_cut = ampls_arr[start_indx:end_indx, :].copy()

    num_f = ampls_arr_cut.shape[0] // 24

    assert num_f == expected_num

    ampls_arr_split = [ampls_arr_cut[ii * 24:(ii + 1) * 24, :] for ii in range(num_f)]

    frame_similarity = np.array([np.linalg.norm(ampls_arr_split[ii] - ampls_arr_split[ii + 1])
                                 for ii in range(num_f - 1)])

    print(frame_similarity)

    reps = np.where(frame_similarity < sim_level)[0]

    if len(reps > 0):
        print('deleting frames', reps)
        for ii in reps:
            ampls_arr_cut[ii * 24:(ii + 1) * 24, :] = np.NaN

    return ampls_arr_cut, diffs, reps
