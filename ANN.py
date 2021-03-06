import numpy as np

def relu(x):
    return np.maximum(0, x)


def relu_d(x):
    return np.maximum(0, np.sign(x))


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
    # log_p = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)]+1e-18)
    # loss = np.sum(log_p) / n_samples
    loss = 0.5 * ((pred-real)**2).sum() / n_samples

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
        ip_dim = w1_0.shape[0]
        hl_dim = w1_0.shape[1]
        op_dim = w2_0.shape[1]
        assert w1_0.shape[1] == w2_0.shape[0]

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
            self.a1 = self.z1
        self.z2 = np.dot(self.a1, self.w2)

        # self.z2 /= 10000

        # self.a2 = softmax(self.z2)

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

        # self.loss = error(self.a2, ys)
        #
        # a2_delta = (self.a2 - ys) / self.batch_size  # for w2
        # z1_delta = np.dot(a2_delta, self.w2.T)
        # if self.nonlinear:
        #     a1_delta = z1_delta * relu_d(self.a1)  # w1
        # else:
        #     a1_delta = z1_delta

        self.loss = error(self.z2, ys)

        a2_delta = (self.z2 - ys)/self.batch_size
        dw2 = np.dot(self.a1.T, a2_delta)

        a1_delta = np.dot(a2_delta, self.w2.T)
        if self.nonlinear == 'relu':
            a1_delta *= relu_d(self.a1)  # w1
        dw1 = np.dot(xs.T, a1_delta)

        adam_dw2, self.m_dw2, self.v_dw2 = self.adam_update(dw2, self.m_dw2, self.v_dw2)
        adam_dw1, self.m_dw1, self.v_dw1 = self.adam_update(dw1, self.m_dw1, self.v_dw1)

        self.w2 -= adam_dw2
        self.w1 -= adam_dw1

        self.t += 1


class DNN_1d:
    def __init__(self, *adam_args, x, y, w1_0, batch_size, num_batches, lr=0.001):
        self.x = x
        self.y = y
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.lr = lr
        self.loss = []
        ip_dim = self.x.shape[1]
        op_dim = self.y.shape[1]

        assert w1_0.shape == (ip_dim, op_dim)
        self.w1 = w1_0

        # parameters for ADAM
        self.m_dw1, self.v_dw1, self.beta1, self.beta2 = adam_args
        self.epsilon = 1e-8
        self.t = 1

    def feedforward(self, z1):

        self.z1 = z1
        self.a1 = softmax(self.z1 * 2.)


    def adam_update(self, dw, m_dw, v_dw):

        t = self.t

        m_dw = self.beta1 * m_dw + (1 - self.beta1) * dw
        v_dw = self.beta2 * v_dw + (1 - self.beta2) * (dw ** 2)

        m_dw_corr = m_dw / (1 - self.beta1 ** t)
        v_dw_corr = v_dw / (1 - self.beta2 ** t)

        adam_dw = self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))

        return adam_dw, m_dw, v_dw

    def backprop(self, xs, ys):

        self.loss = error(self.a1, ys)

        a1_delta = (self.a1 - ys) / self.batch_size

        dw1 = np.dot(xs.T, a1_delta)

        adam_dw1, self.m_dw1, self.v_dw1 = self.adam_update(dw1, self.m_dw1, self.v_dw1)

        self.w1 -= adam_dw1

        self.t += 1


class DNN_MSE:
    def __init__(self, *adam_args, x, y, w1_0, batch_size, num_batches, lr=0.001):
        self.x = x
        self.y = y
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.lr = lr
        self.loss = []
        ip_dim = self.x.shape[1]
        op_dim = self.y.shape[1]

        assert w1_0.shape == (ip_dim, op_dim)
        self.w1 = w1_0
        self.deltas = np.zeros((batch_size, op_dim))

        # parameters for ADAM
        self.m_dw1, self.v_dw1, self.beta1, self.beta2 = adam_args
        self.epsilon = 1e-8
        self.t = 1

    # def feedforward(self, z1):
    #
    #     self.z1 = z1
    #     self.a1 = softmax(self.z1 * 2.)

    def adam_update(self, dw, m_dw, v_dw):

        t = self.t

        m_dw = self.beta1 * m_dw + (1 - self.beta1) * dw
        v_dw = self.beta2 * v_dw + (1 - self.beta2) * (dw ** 2)

        m_dw_corr = m_dw / (1 - self.beta1 ** t)
        v_dw_corr = v_dw / (1 - self.beta2 ** t)

        adam_dw = self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))

        return adam_dw, m_dw, v_dw

    def backprop(self, xs):

        self.loss = (0.5 * self.deltas**2).sum() / self.batch_size

        dw1 = np.dot(xs.T, self.deltas) / self.batch_size

        adam_dw1, self.m_dw1, self.v_dw1 = self.adam_update(dw1, self.m_dw1, self.v_dw1)

        self.w1 -= adam_dw1

        self.t += 1


class DNN_complex:
    def __init__(self, *adam_args, x, y, w1_x_0, w1_y_0, batch_size, num_batches, lr=0.001, scaling=1.):
        self.x = x
        self.y = y
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.lr = lr
        self.loss = []
        self.scaling = scaling
        ip_dim = self.x.shape[1]
        op_dim = self.y.shape[1]

        assert w1_x_0.shape == (ip_dim, op_dim)
        self.w1_x = w1_x_0.copy()
        self.w1_y = w1_y_0.copy()

        # parameters for ADAM
        self.m_dw_x_1, self.v_dw_x_1, self.m_dw_y_1, self.v_dw_y_1, self.beta1, self.beta2 = adam_args
        self.epsilon = 1e-8
        self.t = 1

    def feedforward(self, z1):

        self.z1_x = np.real(z1)
        self.z1_y = np.imag(z1)

        self.a1_x = self.z1_x ** 2
        self.a1_y = self.z1_y ** 2

        self.z2 = self.a1_x + self.a1_y

        self.a2 = softmax(self.z2 * self.scaling)

    def adam_update(self, dw_x, dw_y, m_dw_x, v_dw_x, m_dw_y, v_dw_y):
        t = self.t

        m_dw_x = self.beta1 * m_dw_x + (1 - self.beta1) * dw_x
        v_dw_x = self.beta2 * v_dw_x + (1 - self.beta2) * (dw_x ** 2)

        m_dw_y = self.beta1 * m_dw_y + (1 - self.beta1) * dw_y
        v_dw_y = self.beta2 * v_dw_y + (1 - self.beta2) * (dw_y ** 2)

        m_dw_x_corr = m_dw_x / (1 - self.beta1 ** t)
        v_dw_x_corr = v_dw_x / (1 - self.beta2 ** t)

        m_dw_y_corr = m_dw_y / (1 - self.beta1 ** t)
        v_dw_y_corr = v_dw_y / (1 - self.beta2 ** t)

        adam_dw_x = self.lr * (m_dw_x_corr / (np.sqrt(v_dw_x_corr) + self.epsilon))
        adam_dw_y = self.lr * (m_dw_y_corr / (np.sqrt(v_dw_y_corr) + self.epsilon))

        return adam_dw_x, m_dw_x, v_dw_x, adam_dw_y, m_dw_y, v_dw_y

    def backprop(self, xs, ys):
        self.loss = error(self.a2, ys)

        a2_delta = (self.a2 - ys) / self.batch_size

        a1_delta_x = a2_delta * 2 * self.z1_x
        a1_delta_y = a2_delta * 2 * self.z1_y

        dw_x_1 = np.dot(xs.T, a1_delta_x)
        dw_y_1 = np.dot(xs.T, a1_delta_y)

        adam_dw_x_1, self.m_dw_x_1, self.v_dw_x_1, adam_dw_y_1, self.m_dw_y_1, self.v_dw_y_1 = self.adam_update(dw_x_1,
                                                                                                                dw_y_1,
                                                                                                                self.m_dw_x_1,
                                                                                                                self.v_dw_x_1,
                                                                                                                self.m_dw_y_1,
                                                                                                                self.v_dw_y_1)

        self.w1_x -= adam_dw_x_1
        self.w1_y -= adam_dw_y_1

        self.t += 1