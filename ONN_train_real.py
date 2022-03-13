import time
import numpy as np
from termcolor import colored
from ONN_class import MyONN
from ONN_config import Controller

# network architecture
n = 100
m = 40

# training parameters
batch_size = 240
num_batches = 5
num_epochs = 2
lr = 10e-3

# technical
num_frames = batch_size // 24

R1_ampl = 1.
R2_ampl = 0.2
label_ampl = 0.

scale_guess = 1.4
ref_guess = 6.7

onn_vars = n, m, batch_size, num_frames, scale_guess, ref_guess, \
           R1_ampl, R2_ampl, label_ampl

ONN = MyONN(lr, num_batches, num_epochs, onn_vars)

####################################################

np.random.seed(100)
w1 = np.random.normal(0, 0.5, (n, m))
w1 = np.clip(w1, -ONN.lim_arr, ONN.lim_arr)
np.save('D:/MNIST/data/w1_0.npy', w1)
ONN.w1 = w1

np.random.seed(200)
w2 = np.random.normal(0, 0.5, (m, 1))
np.save('D:/MNIST/data/w2_0.npy', w2)
ONN.w2 = w2

ONN.load_dataset()
ONN.graphs()
ONN.run_calibration()
ONN.init_onn()

####################################################

for epoch_num in range(num_epochs):

    epoch_start_time = time.time()

    ONN.init_epoch(epoch_num)

    for batch_num in range(num_batches):

        ONN.run_batch(batch_num)
        ONN.save_data_batch(epoch_num, batch_num)
        ONN.update_plots_batch()

    ONN.run_validation(epoch_num)
    ONN.save_data_epoch()
    ONN.update_plots_epoch()

    epoch_time = time.time() - epoch_start_time

    print('\n######################################################################')
    print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                  .format(epoch_num, epoch_time, ONN.accs[-1], ONN.loss[-1]), 'green'))
    print('######################################################################\n')

ONN.ctrl.close()
