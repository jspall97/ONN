import time
import sys
import signal
import numpy as np
import cupy as cp
from glumpy import app
from make_dmd_image import make_dmd_rgb, make_dmd_image, make_dmd_rgb_multi
from MNIST import make_dmd_rgb as make_dmd_rgb_new
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm1_image import make_slm_rgb
from multiprocessing import Process, Pipe
from pylon import run_camera, view_camera, Camera, CameraThread
import threading
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])


def keyboardinterrupthandler(signal, frame):
    context.pop()
    print('keyboard interupt - closed')
    exit(0)


signal.signal(signal.SIGINT, keyboardinterrupthandler)


def pad_frames(frs, lead=3, trail=2):
    global null_frame
    frs[0:0] = [null_frame for _ in range(lead)]
    frs.extend([null_frame for _ in range(trail)])
    return frs

def run_frames(frames, fr=0, fc=None, pad=True):
    global cp_arr, target_frames, frame_count, batch_num

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


def update_slm(arrA, arrP):

    gpu_arrA = cp.asarray(arrA)
    gpu_arrP = cp.asarray(arrP)
    img = make_slm_rgb(gpu_arrA, gpu_arrP) #.get()
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
    cam_process = Process(target=run_camera, args=[conn2, 'init'])
    cam_process.daemon = 1
    cam_process.start()
    time.sleep(5)

    batch_num = 0
    null_frame = dmd_one_frame(np.zeros((n, m)))[0]
    null_frames = [null_frame for _ in range(10)]

    conn1.send((0, 100))
    run_frames(frames=null_frames)  # init frames, "batch" 0
    wait_for_sig(conn1)


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

    batch_num = 0

    # ####################
    # # PHASE CALIB #
    # ####################
    #
    #
    ref_col = 62

    ones = np.full((n, m), 1.)
    arr = np.full((n, m), 0.)

    update_slm(ones, arr)

    dmd_col = np.zeros((n, m))
    dmd_col[ref_col, :] = 1.
    dmd_col[55, :] = 1.

    dmd_frame = dmd_one_frame(dmd_col)
    run_frames(frames=dmd_frame, pad=False)

    time.sleep(1)

    view_camera()

    init_cam_dmd_scan()

    dmd_cols = [cp.load('./tools/dmd_imgs/two_cols/col_array_{}.npy'.format(i)) for i in range(5)]
    dmd_cols = pad_frames(dmd_cols)

    phases = np.linspace(0, 2*np.pi, 64)

    # warmup
    for _ in range(5):

        conn1.send((2, 400))

        run_frames(frames=dmd_cols, pad=False)

        wait_for_sig(conn1)

    for indx, p in enumerate(phases):

        print(indx)

        arr = np.full((n, m), 0.)
        arr[ref_col, :] = p
        arr[:, m//2] = 0

        update_slm(ones, arr)

        conn1.send((2+indx, 400))

        run_frames(frames=dmd_cols, pad=False)

        wait_for_sig(conn1)

    conn1.send((1, 0))  # stop camera

    ####################
    # AMPL CALIB #
    ####################
    #
    # ones = np.full((n, m), 1)
    # # ones[:, 0] = 0.05
    #
    # ones[:, m // 2] = 0.05
    #
    # arr = np.full((n, m), 0)
    #
    # update_slm(ones, arr)
    #
    # dmd_col = np.zeros((n, m))
    # # dmd_col[:, 0] = 1
    # dmd_col[60, :] = 1
    #
    # dmd_col[:, m//2] = 1
    #
    # # dmd_col[:, ::2] = 0
    #
    # dmd_frame = dmd_one_frame(dmd_col)
    # run_frames(frames=dmd_frame, pad=False)
    #
    # time.sleep(1)
    #
    # view_camera()
    #
    # init_cam_dmd_scan()
    #
    # dmd_cols = [cp.load('./tools/dmd_imgs/cols/col_array_{}.npy'.format(i)) for i in range(5)]
    # dmd_cols = pad_frames(dmd_cols)
    #
    # conn1.send((2+100, 400))
    #
    # run_frames(frames=dmd_cols, pad=False)
    #
    # wait_for_sig(conn1)
    #
    # conn1.send((1, 0))  # stop camera


    # # warmup
    # for _ in range(5):
    #
    #     conn1.send((2, 400))
    #
    #     run_frames(frames=dmd_cols, pad=False)
    #
    #     wait_for_sig(conn1)
    #
    # ampls = np.linspace(-1, 1, 65)
    #
    # for indx, ampl in enumerate(ampls):
    #
    #     print(indx)
    #
    #     ones = np.full((n, m), ampl)
    #     ones[:, 0] = 0.0
    #     arr = np.full((n, m), 0)
    #
    #     update_slm(ones, arr)
    #
    #     conn1.send((2+indx, 400))
    #
    #     run_frames(frames=dmd_cols, pad=False)
    #
    #     wait_for_sig(conn1)
    #
    # conn1.send((1, 0))  # stop camera