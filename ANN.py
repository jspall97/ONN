import numpy as np
from scipy.io import loadmat
import time
import random

# inputs = loadmat('./tools/MNIST digit - subsampled - 121.mat')
#
# num_train = 60000
# num_test = 10000
#
# trainY_raw = inputs['trainY']
# trainY = np.zeros((num_train, 10))
# for i in range(num_train):
#     trainY[i, trainY_raw[0, i]] = 1
#
# testY_raw = inputs['testY']
# testY = np.zeros((num_test, 10))
# for i in range(num_test):
#     testY[i, testY_raw[0, i]] = 1
#
# trainX_raw = inputs['trainX']
# trainX = np.empty((num_train, 121))
# for i in range(num_train):
#     trainX_k = trainX_raw[i, :] - trainX_raw[i, :].min()
#     trainX_k = trainX_k / trainX_k.max()
#     trainX[i, :] = trainX_k
#
# testX_raw = inputs['testX']
# testX = np.empty((num_test, 121))
# for i in range(num_test):
#     testX_k = testX_raw[i, :] - testX_raw[i, :].min()
#     testX_k = testX_k / testX_k.max()
#     testX[i, :] = testX_k
#
# np.random.seed(0)
# np.random.shuffle(trainX)
# np.random.seed(0)
# np.random.shuffle(trainY)
# np.random.seed(0)
# np.random.shuffle(testX)
# np.random.seed(0)
# np.random.shuffle(testY)
#
# valX = testX[:5000, :].copy()
# testX = testX[5000:, :].copy()
#
# valY = testY[:5000, :].copy()
# testY = testY[5000:, :].copy()
#
# trainX -= 0.1
# trainX = np.clip(trainX, 0, 1)
# trainX /= trainX.max()
#
# valX -= 0.1
# valX = np.clip(valX, 0, 1)
# valX /= valX.max()
#
# testX -= 0.1
# testX = np.clip(testX, 0, 1)
# testX /= testX.max()

# trainX = 1-trainX
# valX = 1-valX
# testX = 1-testX

# trainX = (trainX*dmd_block_w).astype(int)/dmd_block_w
# valX = (valX*dmd_block_w).astype(int)/dmd_block_w
# testX = (testX*dmd_block_w).astype(int)/dmd_block_w

m_act = 24

actual_uppers_arr_1024 = np.load("./tools/actual_uppers_arr_1024.npy")

uppers1_nm = actual_uppers_arr_1024[-1, ...].copy()
uppers1_nm = np.delete(uppers1_nm, [m_act//2], 1)


def relu(x):
    return np.maximum(0, x)


def relu_d(x):
    return np.maximum(0, np.sign(x - 10))


def softmax(x):
    # Numerically stable with large exponentials
    x = x.T
    exps = np.exp(x - x.max(axis=0))
    a2 = exps / np.sum(exps, axis=0)
    a2 = a2.T
    return a2


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def error(pred, real):
    n_samples = real.shape[0]
    log_p = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)]+1e-18)
    loss = np.sum(log_p) / n_samples

    return loss


def accuracy(pred, label):
    correct = (pred == label).astype(int).sum()
    perc = correct * 100 / pred.shape[0]
    return perc

class DNN:
    def __init__(self, *adam_args, x, y, w1_0, w2_0, batch_size, num_batches, lr=0.001, nonlinear=False):
        self.x = x
        self.y = y
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.lr = lr
        self.loss = []
        ip_dim = self.x.shape[1]
        op_dim = self.y.shape[1]
        hl_dim = 24

        self.nonlinear = nonlinear

        assert w1_0.shape == (ip_dim, hl_dim)
        self.w1 = w1_0
        self.w2 = w2_0

        # parameters for ADAM
        self.m_dw1, self.v_dw1, self.m_dw2, self.v_dw2, self.beta1, self.beta2 = adam_args
        self.epsilon = 1e-8
        self.t = 1

    def feedforward(self, z1):

        self.z1 = z1
        if self.nonlinear:
            self.a1 = relu(self.z1)
        else:
            self.a1 = self.z1 #**2
        self.z2 = np.dot(self.a1, self.w2)

        # self.z2 /= 10000

        self.a2 = softmax(self.z2)

        # print(self.z2.min(), self.z2.max())
        # print(self.a2.min(), self.a2.max())


    def adam_update(self, dw, m_dw, v_dw):

        t = self.t

        m_dw = self.beta1 * m_dw + (1 - self.beta1) * dw
        v_dw = self.beta2 * v_dw + (1 - self.beta2) * (dw ** 2)

        m_dw_corr = m_dw / (1 - self.beta1 ** t)
        v_dw_corr = v_dw / (1 - self.beta2 ** t)

        adam_dw = self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))

        return adam_dw, m_dw, v_dw

    def backprop(self, xs, ys):

        self.loss = error(self.a2, ys)

        a2_delta = (self.a2 - ys) / self.batch_size  # for w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        if self.nonlinear:
            a1_delta = z1_delta * relu_d(self.a1)  # w1
        else:
            a1_delta = z1_delta #* 2*self.z1

        dw2 = np.dot(self.a1.T, a2_delta)
        dw1 = np.dot(xs.T, a1_delta)

        adam_dw2, self.m_dw2, self.v_dw2 = self.adam_update(dw2, self.m_dw2, self.v_dw2)
        adam_dw1, self.m_dw1, self.v_dw1 = self.adam_update(dw1, self.m_dw1, self.v_dw1)

        # self.w2 -= adam_dw2
        self.w1 -= adam_dw1

        # self.w1 = np.clip(self.w1, -uppers1_nm, uppers1_nm)
        #
        # self.w1[:, 0] = 0
        # self.w1[:, -1] = 1

        self.t += 1

    def predict(self, x):
        z = np.dot(x, self.w1)
        self.feedforward(z)
        return self.a2.argmax(axis=1)

