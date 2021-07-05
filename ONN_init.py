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


def update_slm(arr, stagger=0):

    gpu_arr = cp.asarray(arr)
    img = make_slm_rgb(gpu_arr, stagger=stagger) #.get()
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
    null_frame = dmd_one_frame(np.ones((n, m)))[0]
    null_frames = [null_frame for _ in range(10)]

    conn1.send((0, 100))
    run_frames(frames=null_frames)  # init frames, "batch" 0
    wait_for_sig(conn1)


if __name__ == '__main__':

    ################
    # SLM display #
    ################

    slm = SLMdisplay(0)
    slm_img = np.full((n, m), 1.)
    update_slm(slm_img)

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

    #####################################
    # adjust brightness for all columns #
    #####################################

    dmd_img = np.ones((n, m))
    # dmd_img[:, 1::2] = 0

    dmd_frame = dmd_one_frame(dmd_img)
    run_frames(frames=dmd_frame, pad=False)

    slm_img = np.full((n, m), 0.1)
    # slm_img[:, 1::2] = 0

    # slm_img = np.full((n, m), 0)
    # slm_img[:, 0] = 1

    update_slm(slm_img, stagger=0)

    # input('press anything to continue')
    #
    # view_camera()

    #######################
    # find spot positions #
    #######################

    calib_ampl = 1

    dmd_img = np.ones((n, m))
    dmd_frame = dmd_one_frame(dmd_img)
    run_frames(frames=dmd_frame, pad=False)

    cam = Camera()

    slm_img = np.full((n, m), calib_ampl)
    slm_img[:, 0::2] = 0
    update_slm(slm_img, stagger=0)
    time.sleep(1)

    cam.capture()
    frame0 = cam.arr.copy().T
    time.sleep(1)

    slm_img = np.full((n, m), calib_ampl)
    slm_img[:, 1::2] = 0
    update_slm(slm_img, stagger=0)
    time.sleep(1)

    cam.capture()
    frame1 = cam.arr.copy().T
    time.sleep(1)

    slm_img = np.full((n, m), calib_ampl)
    update_slm(slm_img, stagger=0)
    time.sleep(1)

    cam.capture()
    frame_both = cam.arr.copy().T
    time.sleep(1)

    np.save('./tools/pylon_img_0.npy', frame0)
    np.save('./tools/pylon_img_1.npy', frame1)
    np.save('./tools/pylon_img_both.npy', frame_both)

    print('saved frames')

    cam.close()

    input('press anything to continue')

    ######################################
    # adjust brightness for single column #
    ######################################

    slm_img = np.full((n, m), 1)
    update_slm(slm_img)

    dmd_col = np.zeros((n, m))
    dmd_col[n//2, :] = 1
    dmd_frame = dmd_one_frame(dmd_col)
    run_frames(frames=dmd_frame, pad=False)

    time.sleep(1)

    view_camera()

    ####################
    # scan all columns #
    ####################

    init_cam_dmd_scan()

    col_arr = cp.load('./tools/dmd_imgs/cols/col_array.npy')
    dmd_cols = [col_arr[i, ...] for i in range(col_arr.shape[0])]

    conn1.send((2, 1000))
    run_frames(frames=dmd_cols)  # init frames, "batch" 0
    wait_for_sig(conn1)

    conn1.send((1, 0))  # stop camera

    ampls_arr = np.load('./tools/init_captures/ampls/batch_0.npy')

    noise_level = 0.5
    diffs = np.diff(ampls_arr[:, m // 2])
    large_diffs = np.where(np.abs(diffs) > noise_level)[0]
    start_indx = large_diffs[0] + 1
    end_indx = start_indx + n

    uppers_nm = ampls_arr[start_indx:end_indx, :]
    np.save('./tools/uppers_nm.npy', uppers_nm)

    breakpoint()

    # init_cam_dmd_scan()
    #
    # uppers_nm = np.empty((n, m))
    #
    # for i in range(n):
    #
    #     dmd_cols = [cp.load('./tools/dmd_imgs/cols/col_array_{}.npy'.format(i)) for _ in range(5)]
    #
    #     conn1.send((i+2, 300))
    #     run_frames(frames=dmd_cols)  # init frames, "batch" 0
    #     wait_for_sig(conn1)
    #
    #     ampls_arr = np.load('./tools/init_captures/ampls/batch_{}.npy'.format(i))
    #
    #     noise_level = 1
    #     diffs = np.diff(ampls_arr[:, m // 2])
    #     large_diffs = np.where(np.abs(diffs) > noise_level)[0]
    #     start_indx = large_diffs[0] + 10
    #     end_indx = start_indx + 30
    #
    #     uppers_nm[i, :] = ampls_arr[start_indx:end_indx, :].mean(axis=0)
    #
    # conn1.send((1, 0))  # stop camera
    #
    # np.save('./tools/uppers_nm.npy', uppers_nm)

    ##################
    # SLM calib scan #
    ##################

    # def slm_calib_scan(calib_ampl, res):
    #
    #     dmd_frame = dmd_one_frame(np.ones((n, m)))
    #     run_frames(frames=dmd_frame, pad=False)
    #
    #     slm_img = np.full((n, m), calib_ampl)
    #     update_slm(slm_img)
    #
    #     view_camera()  # adjust brightness
    #
    #     init_cam_dmd_scan()
    #
    #     for batch_num, ampl in enumerate(np.linspace(0, calib_ampl, res)):
    #
    #         slm_img = np.full((n, m), ampl)
    #         update_slm(slm_img)
    #
    #         conn1.send((2+batch_num, 100))
    #         wait_for_sig(conn1)
    #
    #         print()
    #
    #     conn1.send((1, 0))  # stop camera
    #
    #     uppers_nm = np.load('./tools/uppers_nm.npy')
    #     uppers_norm = uppers_nm/uppers_nm.max()
    #
    #     loaded = np.load('./tools/init_captures/ampls/batch_{}.npy'.format(res-1))
    #     meas = loaded.mean(axis=0)
    #     assert meas.shape[0] == m
    #
    #     compare_indx = np.argmax(meas)
    #     meas_norm_val = meas[compare_indx]
    #     meas = meas * calib_ampl / meas_norm_val
    #
    #     theory = (uppers_norm.copy() * calib_ampl).sum(axis=0)
    #     theory = theory * calib_ampl / theory[compare_indx]
    #
    #     fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))
    #     axs1.set_ylim(0, 1.2)
    #     axs1.plot(theory, linestyle='', marker='o', c='b')
    #     axs1.plot(meas, linestyle='', marker='x', c='r')
    #     plt.draw()
    #
    #     ratio = theory / meas
    #     actual_uppers = uppers_norm.copy() * calib_ampl / ratio
    #
    #     theory_norm_val = actual_uppers.sum(axis=0).max()
    #
    #     theory_new = actual_uppers.sum(axis=0) * calib_ampl / theory_norm_val
    #     fig2, axs2 = plt.subplots(1, 1, figsize=(8, 4))
    #     axs2.set_ylim(0, 1.2)
    #     axs2.plot(theory_new, linestyle='', marker='o', c='b')
    #     axs2.plot(meas, linestyle='', marker='x', c='r')
    #     plt.draw()
    #
    #     plt.show()
    #
    #     actual_uppers_arr = np.empty((n, m, res))
    #
    #     for k, ampl in enumerate(np.linspace(0, calib_ampl, res)):
    #         theory = (actual_uppers.copy() * ampl / calib_ampl).sum(axis=0)
    #         theory = theory / theory_norm_val
    #
    #         loaded = np.load('./tools/init_captures/ampls/batch_{}.npy'.format(k))
    #         meas = loaded.mean(axis=0)
    #         meas = meas / meas_norm_val
    #
    #         ratio = theory / meas
    #
    #         actual_uppers_arr[..., k] = (actual_uppers.copy() * ampl / calib_ampl) / ratio
    #
    #     actual_uppers_arr[..., 0] = 0
    #     actual_uppers_arr = np.maximum.accumulate(actual_uppers_arr, axis=-1)
    #
    #     np.save('./tools/actual_uppers_arr_{}.npy'.format(calib_ampl), actual_uppers_arr)

    def slm_calib_scan(calib_ampl, res):

        dmd_frame = dmd_one_frame(np.ones((n, m)))
        run_frames(frames=dmd_frame, pad=False)

        slm_img = np.full((n, m), calib_ampl)
        update_slm(slm_img)

        view_camera()  # adjust brightness

        init_cam_dmd_scan()

        for batch_num, ampl in enumerate(np.linspace(0, calib_ampl, res)):

            slm_img = np.full((n, m), ampl)
            update_slm(slm_img)

            conn1.send((2+batch_num, 100))
            wait_for_sig(conn1)

            print()

        conn1.send((1, 0))  # stop camera

        uppers_nm = np.load('./tools/uppers_nm.npy')
        uppers_norm = uppers_nm/uppers_nm.max()

        loaded = np.load('./tools/init_captures/ampls/batch_{}.npy'.format(res-1))
        meas_0 = loaded.mean(axis=0)

        indx = np.argmax(meas_0)
        uppers_sum = (uppers_norm.copy()).sum(axis=0)
        uppers_sum_norm = uppers_sum / uppers_sum[indx]

        actual_uppers_arr = np.empty((n, m, res))

        for k, ampl in enumerate(np.linspace(0, calib_ampl, res)):

            loaded = np.load('./tools/init_captures/ampls/batch_{}.npy'.format(k))
            meas = loaded.mean(axis=0)
            meas_norm = meas / meas_0[indx]

            actual_uppers_k = uppers_norm.copy() * meas_norm/uppers_sum_norm

            actual_uppers_arr[..., k] = actual_uppers_k

        actual_uppers_arr[..., 0] = 0
        actual_uppers_arr = np.maximum.accumulate(actual_uppers_arr, axis=-1)

        np.save('./tools/actual_uppers_arr_{}.npy'.format(calib_ampl), actual_uppers_arr)

    slm_calib_scan(1., 64)
    slm_calib_scan(0.2, 24)

    actual_uppers_arr_02 = np.load('./tools/actual_uppers_arr_0.2.npy')
    actual_uppers_arr_1 = np.load('./tools/actual_uppers_arr_1.0.npy')

    x_old = np.concatenate((np.linspace(0, 0.2, 24), np.linspace(0, 1, 64)[13:]))

    y_old1 = actual_uppers_arr_02.copy().transpose(2, 0, 1)
    y_old2 = actual_uppers_arr_1[..., 13:].copy().transpose(2, 0, 1)

    y_old1 = y_old1 * y_old2[0, ...] / y_old1.max(axis=0)

    y_old = np.concatenate((y_old1, y_old2), axis=0)

    f = interp1d(x_old, y_old, kind='linear', fill_value='extrapolate', axis=0)
    xnew = np.linspace(0, 1, 24)
    y_smooth = f(xnew)

    f = interp1d(xnew, y_smooth, kind='linear', fill_value='extrapolate', axis=0)
    xnew = np.linspace(0, 1, 1024)

    actual_uppers_arr_1024 = f(xnew)
    actual_uppers_arr_1024 = np.maximum.accumulate(actual_uppers_arr_1024, axis=0)
    actual_uppers_arr_1024 = np.transpose(actual_uppers_arr_1024, (1, 2, 0))

    actual_uppers_arr_1024 = np.flip(actual_uppers_arr_1024, axis=0)

    xnans = np.where(np.isnan(actual_uppers_arr_1024))[0]
    _, idx = np.unique(xnans, return_index=True)
    xnans = xnans[np.sort(idx)]

    ynans = np.where(np.isnan(actual_uppers_arr_1024))[1]
    _, idx = np.unique(ynans, return_index=True)
    ynans = ynans[np.sort(idx)]

    for k in range(xnans.shape[0]):
        actual_uppers_arr_1024[xnans[k], ynans[k], :] = actual_uppers_arr_1024[xnans[k], ynans[k] - 1, :]

    while xnans.shape[0] > 0:

        xnans = np.where(np.isnan(actual_uppers_arr_1024))[0]
        _, idx = np.unique(xnans, return_index=True)
        xnans = xnans[np.sort(idx)]

        ynans = np.where(np.isnan(actual_uppers_arr_1024))[1]
        _, idx = np.unique(ynans, return_index=True)
        ynans = ynans[np.sort(idx)]

        for k in range(xnans.shape[0]):
            actual_uppers_arr_1024[xnans[k], ynans[k], :] = actual_uppers_arr_1024[xnans[k], ynans[k] - 1, :]

    np.save('./tools/actual_uppers_arr_1024.npy', actual_uppers_arr_1024)

    plt.show()
    context.pop()
    print('closed dmd')
