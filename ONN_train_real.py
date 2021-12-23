import time
import numpy as np
from termcolor import colored
from ONN_class import MyONN
from ONN_config import Controller

# network architecture
n = 100
m = 40
mout = 10
layers = 1
meas_type = 'reals'
is_complex = False
label_block_on = False

# training parameters
batch_size = 240
num_batches = 5
num_epochs = 20
lr = 10e-3

# technical
num_frames = batch_size // 24
ref_spot = m//2
ref_block_val = 0.3
ampl_norm_val = 0.1
scale_guess = 1.4
ref_guess = 6.7
ref_on = False

# search NamedTuple

onn_vars = n, m, ref_spot, ref_block_val, batch_size, num_frames, is_complex, mout, \
          ampl_norm_val, scale_guess, ref_guess, meas_type, layers, ref_on, label_block_on

controller = Controller(*onn_vars)

lim_arr = controller.uppers1_ann.copy()

np.random.seed(100)
w1 = np.random.normal(0, 0.5, (n, mout))
w1 = np.clip(w1, -lim_arr, lim_arr)
np.save('D:/MNIST/data/w1_0.npy', w1)

np.random.seed(200)
w2 = np.random.normal(0, 0.5, (mout, 10))
np.save('D:/MNIST/data/w2_0.npy', w2)

####################################################

ONN = MyONN(controller, w1, w2, lr, num_batches, num_epochs, noise_mean=0., noise_std=0.1, noise_type=None)

ONN.graphs()
ONN.run_calibration()
ONN.init_onn()

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
