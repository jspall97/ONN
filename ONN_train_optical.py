import time
import numpy as np
from ANN import DNN_MSE, accuracy
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
label_block_on = ONN.label_block_on
mout = ONN.mout
ampl_norm_val = ONN.ampl_norm_val
scale_guess = ONN.scale_guess
meas_type = ONN.meas_type

uppers1_nm, uppers1_ann = ONN.load_lut()

layers = 1
ref_on = False

###############
# CALIBRATING #
###############

ONN.init_dmd()

all_deltas = []
all_theories = []

for k in range(5):

    np.random.seed(k)
    slm_arr = np.random.normal(0, 0.4, (n, mout))
    slm_arr = np.clip(slm_arr, -uppers1_ann, uppers1_ann)
    ONN.update_slm(slm_arr, lut=True, ref=ref_on)
    time.sleep(1)

    batch_indxs = np.random.randint(0, 60000, batch_size)

    vecs = ONN.trainX_cp[batch_indxs, :].copy()

    xs = ONN.trainX[batch_indxs, :].copy()
    ys = ONN.trainY[batch_indxs, :].copy()
    labels = ys.argmax(axis=1)

    frames, ampls = ONN.run_batch(vecs, ref=ref_on, labels=labels)

    print(frames.shape)

    if ONN.check_num_frames(ampls, batch_size):

        deltas = ONN.process_ampls(ampls)

        theories = np.dot(xs, slm_arr.copy()) - 2*ys

        all_deltas.append(deltas)
        all_theories.append(theories)

all_deltas = np.array(all_deltas)
all_deltas = all_deltas.reshape(all_deltas.shape[0] * 240, mout)
all_theories = np.array(all_theories)
all_theories = all_theories.reshape(all_theories.shape[0] * 240, mout)

print(all_deltas.shape, all_theories.shape)

np.save('./tools/temp_z1s.npy', all_deltas)
np.save('./tools/temp_theories.npy', all_theories)

norm_params = ONN.find_norm_params(all_theories, all_deltas)

all_deltas = ONN.normalise(all_deltas, norm_params)

error = (all_deltas - all_theories).std()
print(colored('error : {:.3f}'.format(error), 'blue'))
print(colored('signal: {:.3f}'.format(all_theories.std()), 'blue'))

plt.ion()
plt.show()

fig3, [[axs0, axs1, axs2], [axs3, axs4, axs5]] = plt.subplots(2, 3, figsize=(8, 4))

axs3.set_ylim(-10, 10)
axs1.set_ylim(-10, 10)
axs1.set_xlim(-10, 10)
axs5.set_ylim(0, 100)
axs5.set_xlim(0, 30)
axs2.set_ylim(0, 5)
axs2.set_xlim(0, 1500)
axs4.set_ylim(0, 0.5)
axs4.set_xlim(0, 1500)

axs1.plot([-10, 10], [-10, 10], c='black')
eg_line = [axs1.plot(all_theories[:, j], all_deltas[:, j], linestyle='', marker='.', markersize=1)[0]
           for j in range(mout)]

th_line = axs3.plot(all_theories[0, :], linestyle='', marker='o', c='b')[0]
meas_line = axs3.plot(all_deltas[0, :], linestyle='', marker='x', c='r')[0]

img = axs0.imshow(frames[0], aspect='auto')

plt.draw()
plt.pause(0.1)

# breakpoint()

############
# ONN INIT #
############

num_batches = 50
num_epochs = 20

lim_arr = uppers1_ann.copy()

np.random.seed(100)
w1 = np.random.normal(0, 0.5, (n, mout))
w1 = np.clip(w1, -lim_arr, lim_arr)
np.save('D:/MNIST/data/w1_0.npy', w1)

m_dw1 = np.zeros((n, mout))
v_dw1 = np.zeros((n, mout))

beta1 = 0.9
beta2 = 0.999

adam_params = (m_dw1, v_dw1, beta1, beta2)

dnn = DNN_MSE(*adam_params, x=ONN.trainX, y=ONN.trainY, w1_0=w1, batch_size=batch_size,
              num_batches=num_batches, lr=20e-3)

loss = [5]
errors = []
accs = []

axs5.plot(accs, linestyle='-', marker='x', c='g')
axs2.plot(loss, linestyle='-', marker='', c='r')
axs4.plot(errors, linestyle='-', marker='', c='b')

loop_clock = clock.Clock()
loop_clock.tick()

##############
# EPOCH LOOP #
##############

for epoch_num in range(num_epochs):

    epoch_start_time = time.time()

    ONN.update_slm(dnn.w1.copy(), lut=True, ref=ref_on)
    time.sleep(1)

    epoch_rand_indxs = np.arange(60000)
    random.Random(epoch_num).shuffle(epoch_rand_indxs)
    batch_indxs_list = [epoch_rand_indxs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    ONN.init_dmd()

    all_deltas = []
    all_theories = []

    for k in range(3):

        batch_indxs = np.random.randint(0, 60000, batch_size)

        vecs = ONN.trainX_cp[batch_indxs, :]
        xs = ONN.trainX[batch_indxs, :].copy()
        ys = ONN.trainY[batch_indxs, :].copy()
        labels = ys.argmax(axis=1)

        frames, ampls = ONN.run_batch(vecs, ref=ref_on, labels=labels)

        print(ampls.shape)

        if ONN.check_num_frames(ampls, batch_size):
            deltas = ONN.process_ampls(ampls)
            deltas = ONN.normalise(deltas, norm_params)

            theories = np.dot(xs, dnn.w1.copy()) - 2*ys

            all_deltas.append(deltas)
            all_theories.append(theories)

    all_deltas = np.array(all_deltas).reshape(3 * 240, mout)
    all_theories = np.array(all_theories).reshape(3 * 240, mout)

    print(all_deltas.shape)
    print(all_theories.shape)

    norm_params = ONN.update_norm_params(all_theories, all_deltas, norm_params)

    noise_arr = None

    ONN.update_slm(dnn.w1.copy(), lut=True, ref=ref_on, noise_arr_A=noise_arr, noise_arr_phi=None)
    time.sleep(1)

    ##########################

    ############
    # TRAINING #
    ############

    for batch_num in range(num_batches):

        print(batch_num)

        vecs = ONN.trainX_cp[batch_indxs_list[batch_num], :].copy()

        xs = ONN.trainX[batch_indxs_list[batch_num], :].copy()
        ys = ONN.trainY[batch_indxs_list[batch_num], :].copy()
        labels = ys.argmax(axis=1)

        frames, ampls = ONN.run_batch(vecs, ref=ref_on, labels=labels)

        t0 = time.time()

        np.save('D:/MNIST/raw_images/training/images/images_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), frames)
        np.save('D:/MNIST/data/training/ampls/ampls_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), ampls)

        if ONN.check_num_frames(ampls, batch_size):

            deltas = ONN.process_ampls(ampls)
            deltas = ONN.normalise(deltas, norm_params)

            theories = np.dot(xs, dnn.w1.copy()) - 2*ys

            error = (deltas - theories).std()
            print(colored('error : {:.3f}'.format(error), 'blue'))
            errors.append(error)

            dnn.deltas = deltas.copy()
            dnn.backprop(xs)

            dnn.w1 = np.clip(dnn.w1.copy(), -uppers1_ann, uppers1_ann)

            noise_arr = None

            ONN.update_slm(dnn.w1.copy(), lut=True, ref=ref_on, noise_arr_A=noise_arr, noise_arr_phi=None)

            if dnn.loss < loss[-1]:
                best_w1 = dnn.w1.copy()

            loss.append(dnn.loss)
            print(colored('loss : {:.2f}'.format(dnn.loss), 'green'))
            np.save('D:/MNIST/data/loss.npy', np.array(loss))

            new_adam_params = np.array([dnn.m_dw1, dnn.v_dw1, dnn.beta1, dnn.beta2])

            np.save('D:/MNIST/data/adam_params.npy', new_adam_params)

            t4 = time.time()
            print('backprop time: ', t4 - t0)

        else:
            deltas = np.full((batch_size, mout), np.nan)
            theories = np.full((batch_size, mout), np.nan)

            t4 = time.time()

        np.save('D:/MNIST/data/training/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), deltas)
        np.save('D:/MNIST/data/training/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), theories)

        np.save('D:/MNIST/data/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(dnn.w1))

        for j in range(mout):
            eg_line[j].set_xdata(np.real(theories[:, j]))
            eg_line[j].set_ydata(np.real(deltas[:, j]))

        th_line.set_ydata(np.real(theories[0, :]))
        meas_line.set_ydata(np.real(deltas[0, :]))

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

    dnn.w1 = best_w1.copy()
    np.save('D:/MNIST/data/best_w1.npy', best_w1)

    noise_arr = None

    ONN.update_slm(dnn.w1.copy(), lut=True, ref=ref_on, noise_arr_A=noise_arr, noise_arr_phi=None)
    time.sleep(1)

    val_z1s = np.full((4800, mout), np.nan)

    for val_batch_num in range(20):

        print(val_batch_num)

        vecs = ONN.valX_cp[val_batch_num * 240:(val_batch_num + 1) * 240, :].copy()

        frames, ampls = ONN.run_batch(vecs, ref=ref_on, labels=None)

        np.save('D:/MNIST/raw_images/validation/images/images_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), frames)
        np.save('D:/MNIST/data/validation/ampls/ampls_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), ampls)

        xs = ONN.valX[val_batch_num * 240:(val_batch_num + 1) * 240, :].copy()

        if ONN.check_num_frames(ampls, batch_size):
            z1s = ONN.process_ampls(ampls)
            z1s = ONN.normalise(z1s, norm_params)

            val_z1s[val_batch_num * 240:(val_batch_num + 1) * 240, :] = z1s.copy()

            theories = np.dot(xs, dnn.w1.copy())

        else:
            z1s = np.full((batch_size, mout), np.nan)
            theories = np.full((batch_size, mout), np.nan)

        np.save('D:/MNIST/data/validation/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), z1s)
        np.save('D:/MNIST/data/validation/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, val_batch_num), theories)

    mask = ~np.isnan(val_z1s[:, 0])
    val_z1s = val_z1s[mask]
    ys = ONN.valY[:4800].copy()[mask]

    pred = val_z1s.argmax(axis=1)
    label = ys.argmax(axis=1)

    acc = accuracy(pred, label)
    accs.append(acc)

    np.save('D:/MNIST/data/accuracy.npy', np.array(accs))

    axs5.plot(accs, linestyle='-', marker='x', c='g')
    axs2.plot(loss, linestyle='-', marker='', c='r')
    axs4.plot(errors, linestyle='-', marker='', c='b')
    plt.draw()
    plt.pause(0.001)

    epoch_time = time.time() - epoch_start_time

    print('\n######################################################################')
    print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                  .format(epoch_num, epoch_time, accs[-1], loss[-1]), 'green'))
    print('######################################################################\n')

ONN.close()
