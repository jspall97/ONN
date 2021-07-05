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
from ANN import DNN, accuracy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
from termcolor import colored

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])
l = int(dims[2])
dmd_block_w = 15


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
    # time.sleep(0.7)


def wait_for_sig(conn):
    while not conn.poll():
        pass
    conn.recv()


def dmd_one_frame(arr):
    img = make_dmd_image(arr)
    frame = make_dmd_rgb([img for _ in range(24)])
    return [frame]


def init_cam_dmd_scan():
    global conn1, conn2, batch_num

    conn1, conn2 = Pipe()
    cam_process = Process(target=run_camera, args=[conn2, 'train'])
    cam_process.daemon = 1
    cam_process.start()
    time.sleep(5)

    batch_num = 0
    nf = dmd_one_frame(np.ones((n, m)))[0]
    nfs = [nf for _ in range(200)]

    print('INIT CAMERA NULL FRAME')
    conn1.send((0, 500))
    run_frames(frames=nfs, pad=False)  # init frames, "batch" 0
    wait_for_sig(conn1)


def process_frames(ampls_arr, expected_shape, noise_level=2., sim_level=5):

    try:

        # diffs = np.diff(ampls_arr[:, m // 2])
        # large_diffs = np.where(np.abs(diffs) > noise_level)[0]
        # start_indx = large_diffs[0] + 1
        # end_indx = large_diffs[-1] + 1
        # # end_indx = start_indx + 240

        non_null_mask = (ampls_arr[:, m//2] > noise_level)
        non_null_indxs = np.arange(ampls_arr.shape[0])[non_null_mask]
        start_indx = non_null_indxs[0]
        end_indx = non_null_indxs[-1] + 1

        ampls_arr_cut = ampls_arr[start_indx:end_indx, :].copy()

        num_f = ampls_arr_cut.shape[0]

        print(start_indx, end_indx, end_indx - start_indx)

        # fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
        # axs4.plot(ampls_arr[:, m // 2], linestyle='', marker='o', c='b')
        # plt.show()

        if num_f == expected_shape:

            ampls_arr_split = [ampls_arr_cut[ii * 24:(ii + 1) * 24, :] for ii in range(num_f//24)]

            frame_similarity = np.array([np.linalg.norm(ampls_arr_split[ii] - ampls_arr_split[ii + 1])
                                         for ii in range(num_f//24 - 1)])

            print(frame_similarity)

            reps = np.where(frame_similarity < sim_level)[0]

            if len(reps) > 0:
                print('deleting frames', reps)
                for ii in reps:
                    ampls_arr_cut[ii * 24:(ii + 1) * 24, :] = np.NaN

            return ampls_arr_cut, reps, 1

        else:
            print('wrong num frames')
            return None, None, 0

    except Exception as e:
        print(e)
        return None, None, 0




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

    #####################
    # find norm values #
    #####################

    ampl_norm_val = 0.5
    arr = np.full((n, m), ampl_norm_val)
    update_slm(arr, lut=False)

    null_frame = dmd_one_frame(np.zeros((n, m)))[0]
    null_frames = [null_frame for _ in range(20)]

    full_frame = dmd_one_frame(np.ones((n, m)))[0]
    full_frames = [full_frame for _ in range(20)]

    run_frames(frames=full_frames, pad=False)

    # view_camera()

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

    meas_norm_val = meas[indx]

    print(meas)
    print()
    print(theory_norm_val, meas_norm_val)

    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 4))
    axs2.set_ylim(0, 66)
    axs2.plot(theory, linestyle='', marker='o', c='b')
    axs2.plot(meas*theory_norm_val/meas_norm_val, linestyle='', marker='x', c='r')
    plt.draw()

    # plt.show()

    print()
    print('############')
    print()

    # conn1.send((2, 400))  # 0 used for init, 1 used to terminate
    # run_frames(frames=null_frames)
    # wait_for_sig(conn1)

    ###################################
    # define batch and epoch functions#
    ###################################

    def run_batch(batch_num, epoch_num, xs, ys, rand_indxs):
        ''' load frames to display on DMD. start camera recording and display the frames.
        load the save camera images and process to extract the output vectors.
        simulate the rest of the neural network and update weight matrix on slm.
        then save all relevant data for this batch.
        '''

        global batch_frames, loss, w1, dnn

        batch_frames = [cp.load('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/frames/rgb24_{}.npy'.format(i))[0, ...]
                        for i in rand_indxs[batch_num]]

        conn1.send((batch_num + 2, 400))  # 0 used for init, 1 used to terminate
        run_frames(frames=batch_frames)
        wait_for_sig(conn1)
        # time.sleep(1)

        ampls = np.load('./MNIST/pylon_captures/ampls/batch_{}.npy'.format(batch_num))
        print('frame max: ', (ampls**2).max())

        measureds, repeats, success = process_frames(ampls, batch_size, noise_level=1.5)

        if success:

            z1s = measureds * theory_norm_val/meas_norm_val

            mask = ~np.isnan(z1s[:, 0])
            z1s = z1s[mask]
            xss = xs[batch_num][mask]
            yss = ys[batch_num][mask]

            theories = (xss[:, None] * w1.T).transpose(0, 2, 1).sum(axis=1)

            z1s_sign = np.sign(theories)
            z1s *= z1s_sign


            print(measureds.mean(), z1s.mean(), theories.mean())

            dnn.feedforward(z1s)
            dnn.backprop(xss, yss)

            w1 = dnn.w1.copy()
            update_slm(w1, lut=True)

            loss.append(dnn.loss)
            print(colored('loss : {:.2f}'.format(dnn.loss), 'green'))

            np.save('./MNIST/loss.npy', np.array(loss))
            new_adam_params = np.array([dnn.m_dw1, dnn.v_dw1, dnn.m_dw2, dnn.v_dw2, dnn.beta1, dnn.beta2])
            np.save('./MNIST/adam_params.npy', new_adam_params)

        else:
            z1s = np.full((batch_size, m), np.nan)
            theories = np.full((batch_size, m), np.nan)

        np.save('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), z1s)
        np.save('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), theories)

        np.save('./MNIST/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1))
        np.save('./MNIST/w2/w2_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w2))

        return


    def run_validation(epoch_num):

        global batch_frames, dnn, accs

        z1s = np.full((4800, m), np.nan)
        xs = np.load('./MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL/xs.npy')
        ys = np.load('./MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL/ys.npy')

        for val_batch in range(20):

            frs = cp.load('./MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL/frames/rgb24_{}.npy'.format(val_batch))
            batch_frames = [frs[i, ...] for i in range(10)]

            conn1.send((val_batch+2, 400))  # 0 used for init, 1 used to terminate
            run_frames(frames=batch_frames)
            wait_for_sig(conn1)

            ampls = np.load('./MNIST/pylon_captures/ampls/batch_{}.npy'.format(val_batch))

            measureds, repeats, success = process_frames(ampls, 240, noise_level=1.5)

            if success:
                result = measureds * theory_norm_val / meas_norm_val
                print(colored('success', 'green'))
            else:
                print(colored('processing error, skipping', 'red'))
                result = np.nan

            z1s[val_batch * 240:(val_batch + 1) * 240, :] = result

        np.save('./MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL/measured/measured_arr_epoch_{}_raw.npy'
                .format(epoch_num), z1s)

        mask = ~np.isnan(z1s[:, 0])
        z1s = z1s[mask]
        xss = xs[mask]
        yss = ys[mask]

        theories = (xss[:, None] * w1.T).transpose(0, 2, 1).sum(axis=1)

        z1s_sign = np.sign(theories)
        z1s *= z1s_sign

        np.save('./MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL/measured/measured_arr_epoch_{}.npy'
                .format(epoch_num), z1s)
        np.save('./MNIST/valX/SHUFFLE_0_NOREMAP_INDIVIDUAL/theory/theory_arr_epoch_{}.npy'
                .format(epoch_num), theories)

        dnn.feedforward(z1s)

        pred = dnn.a2.argmax(axis=1)
        label = yss.argmax(axis=1)

        acc = accuracy(pred, label)
        accs.append(acc)

        np.save('./MNIST/accuracy.npy', np.array(accs))

        return



    def train_epoch(epoch_num):
        ''' each element in dmd_vecs, xs, ys is a list/array of length batch_size,
        and is accessed by "run_batch()" as foo[batch_num]
        They are generated with list of random indexes, stored in batch_indxs[batch_num].
        Generate these for all batches in the epoch, shuffled differently on each epoch.
        Finally run all batches in the epoch.
        '''

        global epoch_xs, epoch_ys, batch_indxs, batch_frames

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
                    xs_i = np.load('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/xs/xs_{}.npy'.format(indx))
                    ys_i = np.load('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/ys/ys_{}.npy'.format(indx))
                else:
                    xs_i = np.concatenate((xs_i, np.load('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/xs/xs_{}.npy'
                                                         .format(indx))))
                    ys_i = np.concatenate((ys_i, np.load('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/ys/ys_{}.npy'
                                                         .format(indx))))
            epoch_xs.append(xs_i)
            epoch_ys.append(ys_i)

        batch_frames = [cp.load('./MNIST/trainX/SHUFFLE_0_NOREMAP_INDIVIDUAL/frames/rgb24_{}.npy'.format(i))[0, ...]
                        for i in batch_indxs[batch_num]]

        for i in range(num_batches):

            t0 = time.time()

            run_batch(i, epoch_num, epoch_xs, epoch_ys, batch_indxs)

            t1 = time.time()
            print('batch time : {:.2f} '.format(t1 - t0))
            print()

        return


    ###########################
    # loop batches and epochs #
    ###########################

    num_batches = 50
    num_frames = 10
    batch_size = num_frames*24
    num_epochs = 10

    w1 = uppers1_nm.copy() * 0.4 + np.random.normal(0, 0.01, (n, m))
    # w1 = np.random.normal(0, 0.1, (n, m))
    w1 = np.clip(w1, -uppers1_nm, uppers1_nm)
    update_slm(w1, lut=True)

    w2 = np.random.normal(0, 1, (m, l))

    m_dw1 = np.zeros((n, m))
    v_dw1 = np.zeros((n, m))
    m_dw2 = np.zeros((m, l))
    v_dw2 = np.zeros((m, l))
    beta1 = 0.9
    beta2 = 0.999
    adam_params = (m_dw1, v_dw1, m_dw2, v_dw2, beta1, beta2)

    # w1 = np.load('./MNIST/w1/w1_epoch_{}_batch_{}.npy'.format(9, 49))
    # w2 = np.load('./MNIST/w2/w2_epoch_{}_batch_{}.npy'.format(9, 49))
    # adam_params = list(np.load('./MNIST/adam_params.npy', allow_pickle=True))
    # update_slm(w1, lut=True)

    dnn = DNN(*adam_params, w1_0=w1, w2_0=w2, batch_size=batch_size, num_batches=num_batches, lr=1e-3)

    loss = []
    accs = []

    null_frames = [null_frame for _ in range(200)]
    conn1.send((2, 1000))  # 0 used for init, 1 used to terminate
    run_frames(frames=null_frames)
    wait_for_sig(conn1)

    for i in range(num_epochs):

        epoch_t0 = time.time()

        train_epoch(i)

        run_validation(i)

        epoch_t1 = time.time()
        epoch_t = epoch_t1 - epoch_t0

        print('######################################################################')
        print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                      .format(i, epoch_t, accs[-1], loss[-1]), 'green'))
        print('######################################################################\n')

    ###########
    # cleanup #
    ###########

    conn1.send(1)  # stop camera
    context.pop()
    print('closed dmd')















