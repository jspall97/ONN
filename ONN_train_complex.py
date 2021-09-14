import time
import numpy as np
from ANN import DNN_complex, accuracy
import matplotlib.pyplot as plt
import random
from termcolor import colored
from glumpy.app import clock
import ONN_config as ONN

n = ONN.n
m = ONN.m
ref_spot = ONN.ref_spot
ref_block_val = ONN.ref_block_val
batch_size = ONN.batch_size
num_frames = ONN.num_frames
is_complex = ONN.is_complex
mout = ONN.mout
ampl_norm_val = ONN.ampl_norm_val
scale_guess = ONN.scale_guess
meas_type = ONN.meas_type

uppers1_nm, uppers1_ann = ONN.load_lut()

###############
# CALIBRATING #
###############

ONN.init_dmd()

all_z1s = []
all_theories = []

for k in range(5):

    np.random.seed(k)

    slm_arr_X = np.random.normal(0, 0.4, (n, 10))
    slm_arr_X = np.clip(slm_arr_X, -uppers1_ann, uppers1_ann)
    slm_arr_Y = np.random.normal(0, 0.4, (n, 10))
    slm_arr_Y = np.clip(slm_arr_Y, -uppers1_ann, uppers1_ann)

    slm_arr = slm_arr_X + (1j * slm_arr_Y)

    ONN.update_slm(slm_arr, lut=True, ref=False)
    time.sleep(1)

    batch_indxs = np.random.randint(0, 60000, batch_size)

    vecs = ONN.trainX_cp[batch_indxs, :].copy()

    xs = ONN.trainX[batch_indxs, :].copy()

    frames, ampls = ONN.run_batch(vecs, ref=0)

    print(frames.shape)

    if ONN.check_num_frames(ampls, batch_size):

        z1s = ONN.process_ampls(ampls)

        theories = np.dot(xs, slm_arr.copy())

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
all_z1s = ONN.normalise(all_z1s, norm_params)

error_real = np.real(all_z1s - all_theories).std()
error_imag = np.imag(all_z1s - all_theories).std()
print(colored('error real: {:.3f}'.format(error_real), 'blue'))
print(colored('error imag: {:.3f}'.format(error_imag), 'blue'))
print(colored('signal: {:.3f}'.format(all_theories.std()), 'blue'))

plt.ion()
plt.show()

fig3, [[axs0, axs1, axs2], [axs3, axs4, axs5]] = plt.subplots(2, 3, figsize=(8, 4))

axs1.set_ylim(-10, 10)
axs1.set_xlim(-10, 10)

axs2.set_ylim(0, 5)
axs2.set_xlim(0, 1000)

axs3.set_ylim(-10, 10)

axs4.set_ylim(-10, 10)
axs4.set_xlim(-10, 10)

axs5.set_ylim(60, 100)
axs5.set_xlim(0, 20)


axs1.plot([-10, 10], [-10, 10], c='black')
eg_line = [axs1.plot(np.real(all_theories[:, j]), np.real(all_z1s[:, j]), linestyle='', marker='.', markersize=1)[0]
           for j in range(mout)]

th_line = axs3.plot(np.real(all_theories[0, :]), linestyle='', marker='o', c='b')[0]
meas_line = axs3.plot(np.real(all_z1s[0, :]), linestyle='', marker='x', c='r')[0]

axs4.plot([-10, 10], [-10, 10], c='black')
eg_line_imag = [axs4.plot(np.imag(all_theories[:, j]), np.imag(all_z1s[:, j]), linestyle='', marker='.',
                          markersize=1)[0] for j in range(mout)]

th_line_imag = axs3.plot(np.imag(all_theories[0, :]), linestyle='', marker='o', c='g')[0]
meas_line_imag = axs3.plot(np.imag(all_z1s[0, :]), linestyle='', marker='x', c='orange')[0]

img = axs0.imshow(frames[0], aspect='auto')

plt.draw()
plt.pause(0.1)

############
# ONN INIT #
############

num_batches = 50
num_epochs = 20

lim_arr = uppers1_ann.copy()

np.random.seed(100)
w1_x = np.random.normal(0, 0.4, (n, mout))
w1_x = np.clip(w1_x, -lim_arr, lim_arr)
np.random.seed(101)
w1_y = np.random.normal(0, 0.4, (n, mout))
w1_y = np.clip(w1_y, -lim_arr, lim_arr)

np.save('D:/MNIST/data/w1_x_0.npy', w1_x)
np.save('D:/MNIST/data/w1_y_0.npy', w1_y)

m_dw_x_1 = np.zeros((n, mout))
v_dw_x_1 = np.zeros((n, mout))
m_dw_y_1 = np.zeros((n, mout))
v_dw_y_1 = np.zeros((n, mout))

beta1 = 0.9
beta2 = 0.999
adam_params = (m_dw_x_1, v_dw_x_1, m_dw_y_1, v_dw_y_1, beta1, beta2)

dnn = DNN_complex(*adam_params, x=ONN.trainX, y=ONN.trainY, w1_x_0=w1_x, w1_y_0=w1_y,
                  batch_size=batch_size, num_batches=num_batches, lr=10e-3, scaling=.6)

loss = [5]
errors_real = []
errors_imag = []
accs = []

axs5.plot(accs, linestyle='-', marker='x', c='g')
axs2.plot(loss, linestyle='-', marker='', c='r')

loop_clock = clock.Clock()
loop_clock.tick()

##############
# EPOCH LOOP #
##############

for epoch_num in range(num_epochs):

    epoch_start_time = time.time()

    w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())
    ONN.update_slm(w1_z, lut=True, ref=False)
    time.sleep(1)

    epoch_rand_indxs = np.arange(60000)
    random.Random(epoch_num).shuffle(epoch_rand_indxs)
    batch_indxs_list = [epoch_rand_indxs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    ONN.init_dmd()

    all_z1s = []
    all_theories = []

    for k in range(3):

        batch_indxs = np.random.randint(0, 60000, batch_size)

        vecs = ONN.trainX_cp[batch_indxs, :]
        xs = ONN.trainX[batch_indxs, :].copy()

        frames, ampls = ONN.run_batch(vecs, ref=0)

        print(ampls.shape)

        if ONN.check_num_frames(ampls, batch_size):
            z1s = ONN.process_ampls(ampls)

            theories = np.dot(xs, w1_z.copy())

            all_z1s.append(z1s)
            all_theories.append(theories)

    all_z1s = np.array(all_z1s).reshape(3 * 240, mout)
    all_theories = np.array(all_theories).reshape(3 * 240, mout)

    all_z1s = ONN.normalise(all_z1s, norm_params)

    print(all_z1s.shape)
    print(all_theories.shape)

    norm_params = ONN.update_norm_params(all_theories, all_z1s, norm_params)

    # print(norm_params)

    ############
    # TRAINING #
    ############

    for batch_num in range(num_batches):

        print(batch_num)

        vecs = ONN.trainX_cp[batch_indxs_list[batch_num], :].copy()

        frames, ampls = ONN.run_batch(vecs, ref=0)

        t0 = time.time()

        np.save('D:/MNIST/raw_images/training/images/images_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), frames)
        np.save('D:/MNIST/data/training/ampls/ampls_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), ampls)

        xs = ONN.trainX[batch_indxs_list[batch_num], :].copy()
        ys = ONN.trainY[batch_indxs_list[batch_num], :].copy()

        if ONN.check_num_frames(ampls, batch_size):

            z1s = ONN.process_ampls(ampls)
            z1s = ONN.normalise(z1s, norm_params)

            theories = np.dot(xs, w1_z.copy())

            error_real = np.real(z1s - theories).std()
            error_imag = np.imag(z1s - theories).std()
            print(colored('error real: {:.3f}, error imag: {:.3f}'.format(error_real, error_imag), 'blue'))
            errors_real.append(error_real)
            errors_imag.append(error_imag)

            dnn.feedforward(z1s)
            dnn.backprop(xs, ys)

            dnn.w1_x = np.clip(dnn.w1_x.copy(), -uppers1_ann, uppers1_ann)
            dnn.w1_y = np.clip(dnn.w1_y.copy(), -uppers1_ann, uppers1_ann)

            ampl_noise = np.random.normal(0, 0.3, (n, m))

            w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())

            ONN.update_slm(w1_z, lut=True, ref=False, noise_arr_A=None, noise_arr_phi=None)

            if dnn.loss < loss[-1]:
                best_w1_x = dnn.w1_x.copy()
                best_w1_y = dnn.w1_y.copy()

            loss.append(dnn.loss)
            print(colored('loss : {:.2f}'.format(dnn.loss), 'green'))
            np.save('D:/MNIST/data/loss.npy', np.array(loss))

            new_adam_params = np.array([dnn.m_dw_x_1, dnn.v_dw_x_1, dnn.m_dw_y_1,
                                        dnn.v_dw_y_1, dnn.beta1, dnn.beta2])

            np.save('D:/MNIST/data/adam_params.npy', new_adam_params)

            t4 = time.time()
            print('backprop time: ', t4 - t0)

        else:

            z1s = np.full((batch_size, mout), np.nan + (1j * np.nan))
            theories = np.full((batch_size, mout), np.nan + (1j * np.nan))

            t4 = time.time()

        np.save('D:/MNIST/data/training/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), z1s)
        np.save('D:/MNIST/data/training/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), theories)

        np.save('D:/MNIST/data/w1/w1_x_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1_x))
        np.save('D:/MNIST/data/w1/w1_y_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1_y))

        for j in range(mout):
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
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.001)

        t5 = time.time()
        print('save/plot time: ', t5 - t4)

        dt = loop_clock.tick()
        print(colored(dt, 'yellow'))
        print()

    ##############
    # VALIDATION #
    ##############

    dnn.w1_x = best_w1_x.copy()
    dnn.w1_y = best_w1_y.copy()

    np.save('D:/MNIST/data/best_w1_x.npy', best_w1_x)
    np.save('D:/MNIST/data/best_w1_y.npy', best_w1_y)

    ampl_noise = np.random.normal(0, 0.3, (n, m))

    w1_z = dnn.w1_x.copy() + (1j * dnn.w1_y.copy())

    ONN.update_slm(w1_z, lut=True, ref=False, noise_arr_A=None, noise_arr_phi=None)
    time.sleep(1)

    val_z1s = np.full((4800, 10), np.nan + (1j * np.nan))

    for val_batch_num in range(20):

        print(val_batch_num)

        vecs = ONN.valX_cp[val_batch_num * 240:(val_batch_num + 1) * 240, :].copy()

        frames, ampls = ONN.run_batch(vecs, ref=0)

        np.save('D:/MNIST/raw_images/validation/images/images_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), frames)
        np.save('D:/MNIST/data/validation/ampls/ampls_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), ampls)

        xs = ONN.valX[val_batch_num * 240:(val_batch_num + 1) * 240, :].copy()

        if ONN.check_num_frames(ampls, batch_size):
            z1s = ONN.process_ampls(ampls)
            z1s = ONN.normalise(z1s, norm_params)

            val_z1s[val_batch_num * 240:(val_batch_num + 1) * 240, :] = z1s.copy()

            theories = np.dot(xs, w1_z.copy())

        else:
            z1s = np.full((batch_size, mout), np.nan + (1j * np.nan))
            theories = np.full((batch_size, mout), np.nan + (1j * np.nan))

        np.save('D:/MNIST/data/validation/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), z1s)
        np.save('D:/MNIST/data/validation/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), theories)

    mask = ~np.isnan(np.real(val_z1s)[:, 0])
    val_z1s = val_z1s[mask]
    ys = ONN.valY[:4800].copy()[mask]

    dnn.feedforward(val_z1s)

    pred = dnn.a2.argmax(axis=1)
    label = ys.argmax(axis=1)

    acc = accuracy(pred, label)
    accs.append(acc)

    np.save('D:/MNIST/data/accuracy.npy', np.array(accs))

    axs5.plot(accs, linestyle='-', marker='x', c='g')
    axs2.plot(loss, linestyle='-', marker='', c='r')
    plt.draw()
    plt.pause(0.001)

    epoch_time = time.time() - epoch_start_time

    print('\n######################################################################')
    print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                  .format(epoch_num, epoch_time, accs[-1], loss[-1]), 'green'))
    print('######################################################################\n')

ONN.close()
