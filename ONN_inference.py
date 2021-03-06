import time
import numpy as np
import cupy as cp
from termcolor import colored
import matplotlib.pyplot as plt

import ONN_config as ONN
from ANN import accuracy, softmax

n = ONN.n
m = ONN.m
ref_spot = ONN.ref_spot
ref_block_val = ONN.ref_block_val
batch_size = ONN.batch_size
num_frames = ONN.num_frames
is_complex = ONN.is_complex
label_block_on = ONN.label_block_on
mout = ONN.mout
ampl_norm_val = ONN.ampl_norm_val
scale_guess = ONN.scale_guess
meas_type = ONN.meas_type

uppers1_nm, uppers1_ann = ONN.load_lut()

layers = 1
ref_on = False

if meas_type == 'reals':
    # w1 = np.load('D:/MNIST/data/best_w1_offline.npy')

    w1 = np.load('D:/MNIST/data/best_w1.npy')

    ONN.update_slm(w1, lut=True, ref=ref_on)
    time.sleep(1)

    if layers == 2:
        # w2 = np.load('D:/MNIST/data/best_w2_offline.npy')
        w2 = np.load('D:/MNIST/data/best_w2.npy')

else:
    w1_x = np.load('D:/MNIST/data/best_w1_x.npy')
    w1_y = np.load('D:/MNIST/data/best_w1_y.npy')

    # w1_x = np.load('D:/MNIST/data/best_w1_x_offline.npy')
    # w1_y = np.load('D:/MNIST/data/best_w1_y_offline.npy')

    w1 = w1_x.copy() + (1j * w1_y.copy())

    ONN.update_slm(w1, lut=True, ref=ref_on)
    time.sleep(1)

accs = []

for rep in range(3):

    ###############
    # CALIBRATING #
    ###############

    ONN.init_dmd()

    all_z1s = []
    all_theories = []

    for k in range(5):

        batch_indxs = np.random.randint(0, 5000, batch_size)

        vecs = ONN.testX_cp[batch_indxs, :]

        xs = ONN.testX[batch_indxs, :].copy()

        frames, ampls = ONN.run_batch(vecs, ref=ref_on)

        print(frames.shape)

        if ONN.check_num_frames(ampls, batch_size):

            z1s = ONN.process_ampls(ampls)

            theories = np.dot(xs, w1.copy())
            if meas_type == 'complex_power':
                theories = np.abs(theories) ** 2

            all_z1s.append(z1s)
            all_theories.append(theories)

    all_z1s = np.array(all_z1s)
    all_z1s = all_z1s.reshape(all_z1s.shape[0] * 240, mout)
    all_theories = np.array(all_theories)
    all_theories = all_theories.reshape(all_theories.shape[0] * 240, mout)

    print(all_z1s.shape, all_theories.shape)

    np.save('./tools/temp_z1s.npy', all_z1s)
    np.save('./tools/temp_theories.npy', all_theories)

    norm_params = ONN.find_norm_params(all_theories, all_z1s)

    print(norm_params)

    all_z1s = ONN.normalise(all_z1s, norm_params)

    if meas_type == 'complex':
        error_real = np.real(all_z1s - all_theories).std()
        error_imag = np.imag(all_z1s - all_theories).std()
        print(colored('error real: {:.3f}'.format(error_real), 'blue'))
        print(colored('error imag: {:.3f}'.format(error_imag), 'blue'))

    else:
        error = (all_z1s - all_theories).std()
        print(colored('error : {:.3f}'.format(error), 'blue'))

    print(colored('signal: {:.3f}'.format(all_theories.std()), 'blue'))

    if meas_type == 'complex_power':
        lower = 0
        upper = 80
    else:
        lower = -10
        upper = 10

    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 4))
    axs3.set_xlim(lower, upper)
    axs3.set_ylim(lower, upper)
    axs3.plot([lower, upper], [lower, upper], c='black')
    for j in range(mout):
        axs3.plot(all_theories[:, j], all_z1s[:, j], linestyle='', marker='.', markersize=1)
    plt.draw()

    fig4, axs4 = plt.subplots(1, 1, figsize=(8, 4))
    axs4.set_ylim(lower, upper)
    axs4.plot(all_theories[0, :], linestyle='', marker='o', c='b')
    axs4.plot(all_z1s[0, :], linestyle='', marker='x', c='r')
    plt.draw()

    plt.show()

    ###########
    # TESTING #
    ###########

    start_time = time.time()

    ONN.init_dmd()

    if meas_type == 'complex':
        test_z1s = np.full((5000, mout), np.nan+(1j*np.nan))
        test_theories = np.full((5000, mout), np.nan+(1j*np.nan))
    else:
        test_z1s = np.full((5000, mout), np.nan)
        test_theories = np.full((5000, mout), np.nan)

    for test_batch_num in range(20):

        print(test_batch_num)
        vecs = ONN.testX_cp[test_batch_num * 240:(test_batch_num + 1) * 240, :].copy()

        frames, ampls = ONN.run_batch(vecs, ref=ref_on)

        np.save('D:/MNIST/raw_images/testing/images/images_batch_{}.npy'
                .format(test_batch_num), frames)
        np.save('D:/MNIST/data/testing/ampls/ampls_rep_{}_batch_{}.npy'
                .format(rep, test_batch_num), ampls)

        xs = ONN.testX[test_batch_num * 240:(test_batch_num + 1) * 240, :].copy()

        if ONN.check_num_frames(ampls, batch_size):

            z1s = ONN.process_ampls(ampls)
            z1s = ONN.normalise(z1s, norm_params)

            theories = np.dot(xs, w1.copy())
            if meas_type == 'complex_power':
                theories = np.abs(theories) ** 2

            test_z1s[test_batch_num * 240:(test_batch_num + 1) * 240, :] = z1s.copy()
            test_theories[test_batch_num * 240:(test_batch_num + 1) * 240, :] = theories.copy()

        else:
            z1s = np.full((batch_size, mout), np.nan+(1j*np.nan))
            theories = np.full((batch_size, mout), np.nan+(1j*np.nan))

        np.save('D:/MNIST/data/testing/measured/measured_arr_rep_{}_batch_{}.npy'
                .format(rep, test_batch_num), z1s)
        np.save('D:/MNIST/data/testing/theory/theory_arr_rep_{}_batch_{}.npy'
                .format(rep, test_batch_num), theories)

    test_batch_num = 20

    print(test_batch_num)

    vecs = cp.zeros((240, n))
    vecs[:200, :] = ONN.testX_cp[4800:, :].copy()

    frames, ampls = ONN.run_batch(vecs, ref=ref_on)

    if ampls.shape[0] == 240:
        ampls = ampls[:200, :]

    np.save('D:/MNIST/raw_images/testing/images/images_batch_{}.npy'
            .format(test_batch_num), frames)
    np.save('D:/MNIST/data/testing/ampls/ampls_rep_{}_batch_{}.npy'
            .format(rep, test_batch_num), ampls)

    xs = ONN.testX[4800:, :].copy()

    if ONN.check_num_frames(ampls, 200):

        z1s = ONN.process_ampls(ampls)
        z1s = ONN.normalise(z1s, norm_params)

        theories = np.dot(xs, w1.copy())
        if meas_type == 'complex_power':
            theories = np.abs(theories) ** 2

        test_z1s[4800:, :] = z1s.copy()
        test_theories[4800:, :] = theories.copy()

    else:
        if meas_type == 'complex':
            z1s = np.full((200, mout), np.nan+(1j*np.nan))
            theories = np.full((200, mout), np.nan+(1j*np.nan))
        else:
            z1s = np.full((200, mout), np.nan)
            theories = np.full((200, mout), np.nan)

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

    if meas_type == 'reals':

        if label_block_on:
            pred = test_z1s.argmax(axis=1)

        else:
            if layers == 1:

                a1s = softmax(test_z1s*2.)
                pred = a1s.argmax(axis=1)

            elif layers == 2:

                def relu(x):
                    return np.maximum(0, x)

                a1s = relu(test_z1s)
                z2s = np.dot(a1s, w2)
                a2s = softmax(z2s)
                pred = a2s.argmax(axis=1)

    elif meas_type == 'complex':
        z1_x = np.real(test_z1s.copy())
        z1_y = np.imag(test_z1s.copy())

        # z1_x = np.real(test_theories.copy())
        # z1_y = np.imag(test_theories.copy())

        z2 = (z1_x ** 2) + (z1_y ** 2)
        scaling = 0.6
        a2 = softmax(z2 * scaling)
        pred = a2.argmax(axis=1)

    elif meas_type == 'complex_power':
        print('complex power')
        z2 = test_z1s.copy()
        # z2 = test_theories.copy()
        scaling = 0.6
        a2 = softmax(z2 * scaling)
        pred = a2.argmax(axis=1)

    ys = ONN.testY.copy()[mask]
    label = ys.argmax(axis=1)

    acc = accuracy(pred, label)

    accs.append(acc)

    print('\n###########################################')
    print(colored('time : {:.2f}, accuracy : {:.2f}'.format(time.time() - start_time, acc), 'green'))
    print('###########################################\n')

    print()

    time.sleep(1)

print(np.mean(np.array(accs)))

ONN.close()
