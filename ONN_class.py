from ANN import DNN, DNN_1d, accuracy
import matplotlib.pyplot as plt
import random
from termcolor import colored
import time
import numpy as np
import cupy as cp
from scipy.io import loadmat
from termcolor import colored
from glumpy.app import clock
from ONN_config import Controller


class MyONN:

    def __init__(self, lr, num_batches, num_epochs, onn_vars):

        self.ctrl = Controller(*onn_vars)

        self.n = self.ctrl.n
        self.m = self.ctrl.m
        self.batch_size = self.ctrl.batch_size
        self.num_frames = self.ctrl.num_frames

        self.w1 = np.empty((self.n, self.m), dtype=np.uint32)
        self.w2 = np.empty((self.m, 1), dtype=np.uint32)
        self.lr = lr
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.loop_clock = clock.Clock()

        self.accs = []
        self.loss = []
        self.errors1 = []
        self.errors2 = []
        self.best_w1 = None
        self.best_w2 = None

        self.lim_arr = self.ctrl.uppers1_nm

        self.dnn = None

        self.trainX = None
        self.valX = None
        self.testX = None
        self.trainX_cp = None
        self.valX_cp = None
        self.testX_cp = None
        self.trainY = None
        self.valY = None
        self.testY = None

        self.batch_indxs_list = []

        self.measured = None
        self.theory = None

    def load_dataset(self):

        num_train = 60000
        num_test = 10000

        self.trainX = (trainX * self.ctrl.dmd_block_w).astype(int) / self.ctrl.dmd_block_w
        self.valX = (valX * self.ctrl.dmd_block_w).astype(int) / self.ctrl.dmd_block_w
        self.testX = (testX * self.ctrl.dmd_block_w).astype(int) / self.ctrl.dmd_block_w

        self.trainX_cp = cp.array(trainX, dtype=cp.float32)
        self.valX_cp = cp.array(valX, dtype=cp.float32)
        self.testX_cp = cp.array(testX, dtype=cp.float32)

    def graphs(self):

        plt.ion()
        plt.show()

        self.fig3, [[self.axs0, self.axs1, self.axs2],
                    [self.axs3, self.axs4, self.axs5]] = plt.subplots(2, 3, figsize=(8, 4))

        self.axs3.set_ylim(-10, 10)
        self.axs1.set_ylim(-10, 10)
        self.axs1.set_xlim(-10, 10)
        self.axs5.set_ylim(0, 100)
        self.axs5.set_xlim(0, 30)
        self.axs2.set_ylim(0, 5)
        self.axs2.set_xlim(0, 1500)
        self.axs4.set_ylim(0, 0.5)
        self.axs4.set_xlim(0, 1500)

        self.axs1.plot([-10, 10], [-10, 10], c='black')

        self.eg_line = [self.axs1.plot(np.zeros(self.ctrl.batch_size), np.zeros(self.ctrl.batch_size),
                                       linestyle='', marker='.', markersize=1)[0] for _ in range(self.ctrl.m)]

        self.th_line = self.axs3.plot(np.zeros(self.ctrl.m), linestyle='', marker='o', c='b')[0]
        self.meas_line = self.axs3.plot(np.zeros(self.ctrl.m), linestyle='', marker='x', c='r')[0]

        self.img = self.axs0.imshow(np.zeros((140, 672)), aspect='auto', vmin=0, vmax=255)

        plt.draw()
        plt.pause(0.1)

    def run_calibration(self):

        measured1 = []
        theory1 = []
        measured2 = []
        theory2 = []

        passed = 0
        failed = 0
        while passed < 5:

            np.random.seed(passed)
            slm1_arr = np.random.normal(0, 0.4, (self.n, self.m))
            slm1_arr = np.clip(slm1_arr, -self.lim_arr, self.lim_arr)

            slm2_arr = np.random.normal(0, 0.4, (self.m, 1))

            self.ctrl.update_slm1(slm1_arr, lut=True)
            self.ctrl.update_slm2(slm2_arr, lut=True)

            time.sleep(1)

            batch_indxs = np.random.randint(0, 60000, self.batch_size)

            vecs = self.trainX_cp[batch_indxs, :].copy()
            xs = self.trainX[batch_indxs, :].copy()

            success = self.ctrl.run_batch(vecs, normalisation=False)

            if success:
                _measured = self.ctrl.z1s.copy()
                _theory = np.dot(xs, slm1_arr.copy())
                measured1.append(_measured)
                theory1.append(_theory)

                _measured = self.ctrl.z2s.copy()
                _theory = np.dot(_theory, slm2_arr.copy())
                measured2.append(_measured)
                theory2.append(_theory)

                passed += 1

            else:
                failed += 1

            if failed > 3:
                raise TimeoutError

        measured1 = np.array(measured1).reshape(5 * self.ctrl.batch_size, self.m)
        theory1 = np.array(theory1).reshape(5 * self.ctrl.batch_size, self.m)
        measured2 = np.array(measured2).reshape(5 * self.ctrl.batch_size, 1)
        theory2 = np.array(theory2).reshape(5 * self.ctrl.batch_size, 1)

        np.save('./tools/temp_z1s.npy', measured1)
        np.save('./tools/temp_theories1.npy', theory1)
        np.save('./tools/temp_z2s.npy', measured2)
        np.save('./tools/temp_theories2.npy', theory2)

        self.ctrl.find_norm_params(theory1, measured1, theory2, measured2)
        self.ctrl.process_ampls(success=True, normalise=True)

        error1 = (self.ctrl.z1s - theory1).std()
        error2 = (self.ctrl.z2s - theory2).std()
        print(colored(f'error1 : {error1:.3f}, error2 : {error2:.3f}', 'blue'))
        print(colored(f'signal1 : {theory1.std():.3f}, signal2 : {theory2.std():.3f}', 'blue'))

    def init_onn(self):

        m_dw1 = np.zeros((self.ctrl.n, self.ctrl.m))
        v_dw1 = np.zeros((self.ctrl.n, self.ctrl.m))

        beta1 = 0.9
        beta2 = 0.999

        m_dw2 = np.zeros((self.ctrl.m, 1))
        v_dw2 = np.zeros((self.ctrl.m, 1))

        adam_params = (m_dw1, v_dw1, m_dw2, v_dw2, beta1, beta2)

        self.dnn = DNN(*adam_params, x=self.trainX, y=self.trainY, w1_0=self.w1, w2_0=self.w2,
                       batch_size=self.batch_size, num_batches=self.num_batches, lr=self.lr,
                       nonlinear=True)

        self.axs5.plot(self.accs, linestyle='-', marker='x', c='g')
        self.axs2.plot(self.loss, linestyle='-', marker='', c='r')
        self.axs4.plot(self.errors1, linestyle='-', marker='', c='b')
        self.axs4.plot(self.errors2, linestyle='-', marker='', c='g')

        self.loop_clock.tick()

    def init_epoch(self, epoch_num):

        self.ctrl.update_slm1(self.w1.copy(), lut=True)
        self.ctrl.update_slm2(self.w2.copy(), lut=True)
        time.sleep(1)

        epoch_rand_indxs = np.arange(60000)
        random.Random(epoch_num).shuffle(epoch_rand_indxs)
        self.batch_indxs_list = [epoch_rand_indxs[i * self.ctrl.batch_size: (i + 1) * self.ctrl.batch_size]
                                 for i in range(self.num_batches)]

        self.ctrl.init_dmd()

        all_z1s = []
        all_theories1 = []
        all_z2s = []
        all_theories2 = []

        passed = 0
        failed = 0
        while passed < 3:

            batch_indxs = np.random.randint(0, 60000, self.batch_size)

            vecs = self.trainX_cp[batch_indxs, :].copy()
            xs = self.trainX[batch_indxs, :].copy()

            success = self.ctrl.run_batch(vecs, normalisation=True)

            if success:
                theories1 = np.dot(xs, self.dnn.w1.copy())
                theories2 = np.dot(theories1, self.dnn.w2.copy())
                all_z1s.append(self.ctrl.z1s)
                all_theories1.append(theories1)
                all_z2s.append(self.ctrl.z2s)
                all_theories2.append(theories2)
                passed += 1
            else:
                failed += 1

            if failed > 3:
                raise TimeoutError

        all_z1s = np.array(all_z1s).reshape(3 * self.batch_size, self.m)
        all_theories1 = np.array(all_theories1).reshape(3 * self.batch_size, self.m)
        all_z2s = np.array(all_z2s).reshape(3 * self.batch_size, 1)
        all_theories2 = np.array(all_theories2).reshape(3 * self.batch_size, 1)

        self.ctrl.update_norm_params(all_theories1, all_z1s, all_theories2, all_z2s)

        self.ctrl.update_slm1(self.dnn.w1.copy(), lut=True)
        self.ctrl.update_slm2(self.dnn.w2.copy(), lut=True)

        time.sleep(1)

    def run_batch(self, batch_num):

        print(batch_num)

        vecs = self.trainX_cp[self.batch_indxs_list[batch_num], :].copy()

        success = self.ctrl.run_batch(vecs, normalisation=True)

        t0 = time.perf_counter()

        xs = self.trainX[self.batch_indxs_list[batch_num], :].copy()
        ys = self.trainY[self.batch_indxs_list[batch_num], :].copy()

        if success:
            measured = self.ctrl.z1s.copy()
            theory = np.dot(xs, self.dnn.w1.copy())

            error = (measured - theory).std()
            self.errors.append(error)
            print(colored('error : {:.3f}'.format(error), 'blue'))

            self.dnn.feedforward(measured)
            self.dnn.backprop(xs, ys)

            self.dnn.w1 = np.clip(self.dnn.w1.copy(), -self.lim_arr, self.lim_arr)

            self.ctrl.update_slms(self.dnn.w1.copy(), self.dnn.w2.copy(),
                                  slm1_lut=True, slm2_lut=True)
            # noise_arr_A=self.noise_arr(), noise_arr_phi=None)

            if self.dnn.loss < self.loss[-1]:
                self.best_w1 = self.dnn.w1.copy()
                self.best_w2 = self.dnn.w2.copy()

            self.loss.append(self.dnn.loss)
            print(colored('loss : {:.2f}'.format(self.dnn.loss), 'green'))

        else:
            measured = np.full((self.ctrl.batch_size, self.ctrl.m), np.nan)
            theory = np.full((self.ctrl.batch_size, self.ctrl.m), np.nan)

        self.measured = measured
        self.theory = theory

        t1 = time.perf_counter()
        print('backprop time: {:.2f}'.format(t1 - t0))

        dt = self.loop_clock.tick()
        print(colored(dt, 'yellow'))
        print()

    def run_validation(self, epoch_num):

        self.dnn.w1 = self.best_w1.copy()
        self.dnn.w2 = self.best_w2.copy()

        self.ctrl.update_slms(self.dnn.w1.copy(), self.dnn.w2.copy(),
                              slm1_lut=True, slm2_lut=True)
        # noise_arr_A=self.noise_arr(), noise_arr_phi=None)
        time.sleep(1)

        measured = []
        theory = []

        val_batch_num = 0
        failed = 0
        while val_batch_num < 20:

            print(val_batch_num)

            xs = self.valX[val_batch_num * self.ctrl.batch_size:(val_batch_num + 1) * self.ctrl.batch_size, :].copy()
            vecs = self.valX_cp[val_batch_num * self.ctrl.batch_size:(val_batch_num + 1) * self.ctrl.batch_size, :].copy()

            success = self.ctrl.run_batch(vecs, normalise=True)

            if success:
                _measured = self.ctrl.z1s.copy()
                _theory = np.dot(xs, self.dnn.w1.copy())
                next_batch = True
            else:
                if failed > 3:
                    _measured = np.full((self.ctrl.batch_size, self.ctrl.m), np.nan)
                    _theory = np.full((self.ctrl.batch_size, self.ctrl.m), np.nan)
                    next_batch = True
                    failed = 0
                else:
                    failed += 1
                    next_batch = False
                    print("retrying batch")

            if next_batch:

                measured.append(_measured)
                theory.append(_theory)

                np.save('D:/MNIST/raw_images/validation/images/images_epoch_{}_batch_{}.npy'
                        .format(epoch_num, val_batch_num), self.ctrl.frames)
                np.save('D:/MNIST/data/validation/ampls/ampls_epoch_{}_batch_{}.npy'
                        .format(epoch_num, val_batch_num), self.ctrl.ampls)
                np.save('D:/MNIST/data/validation/measured/measured_arr_epoch_{}_batch_{}.npy'
                        .format(epoch_num, val_batch_num), _measured)
                np.save('D:/MNIST/data/validation/theory/theory_arr_epoch_{}_batch_{}.npy'
                        .format(epoch_num, val_batch_num), _theory)

                val_batch_num += 1

        measured = np.array(measured).reshape((4800, self.ctrl.m))
        mask = ~np.isnan(measured[:, 0])
        measured = measured[mask]
        ys = self.valY[:4800].copy()[mask]

        self.dnn.feedforward(measured)

        pred = self.dnn.z2[:, 0]
        label = ys[:, 0]

        acc = accuracy(pred, label)
        self.accs.append(acc)

    def save_data_batch(self, epoch_num, batch_num):

        np.save('D:/MNIST/raw_images/training/images/images_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.ctrl.frames)
        np.save('D:/MNIST/data/training/ampls/ampls_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.ctrl.ampls)

        np.save('D:/MNIST/data/loss.npy', np.array(self.loss))

        np.save('D:/MNIST/data/training/measured/measured_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.measured)
        np.save('D:/MNIST/data/training/theory/theory_arr_epoch_{}_batch_{}.npy'
                .format(epoch_num, batch_num), self.theory)

        np.save('D:/MNIST/data/w1/w1_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(self.dnn.w1))
        np.save('D:/MNIST/data/w2/w2_epoch_{}_batch_{}.npy'.format(epoch_num, batch_num), np.array(self.dnn.w2))

    def save_data_epoch(self):

        np.save('D:/MNIST/data/accuracy.npy', np.array(self.accs))
        np.save('D:/MNIST/data/best_w1.npy', self.best_w1)
        np.save('D:/MNIST/data/best_w2.npy', self.best_w2)

    def update_plots_batch(self):

        for j in range(self.ctrl.m):
            self.eg_line[j].set_xdata(np.real(self.theory[:, j]))
            self.eg_line[j].set_ydata(np.real(self.measured[:, j]))

        self.th_line.set_ydata(np.real(self.theory[0, :]))
        self.meas_line.set_ydata(np.real(self.measured[0, :]))

        if len(self.ctrl.frames) > 0:
            self.img.set_array(self.ctrl.frames[0])

        plt.draw()
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.001)

    def update_plots_epoch(self):

        self.axs5.plot(self.accs, linestyle='-', marker='x', c='g')
        self.axs2.plot(self.loss, linestyle='-', marker='', c='r')
        self.axs4.plot(self.errors, linestyle='-', marker='', c='b')
        plt.draw()
        plt.pause(0.001)
