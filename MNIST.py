import time
import sys
import signal
import numpy as np
import cupy as cp
from scipy.io import loadmat
import random

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])

######################################################
# SLM1 (Santec) #
######################################################

slm_w = 1406
slm_h = 774

slm_block_w = int(((slm_w + 1) / n))
slm_block_h = int(((slm_h + 1) / m))

g1_x = (slm_w - (slm_block_w * n)) // (n - 1)
g1_y = (slm_h - (slm_block_h * m)) // (m - 1)

exl = (slm_w - (slm_block_w * n) - (n - 1) * g1_x) // 2
exr = (slm_w - (slm_block_w * n) - (n - 1) * g1_x) - exl
eyt = (slm_h - (slm_block_h * m) - (m - 1) * g1_y) // 2
eyb = (slm_h - (slm_block_h * m) - (m - 1) * g1_y) - eyt + 1

slm_centres_x = np.array([exl + (i * (slm_block_w + g1_x)) + (slm_block_w // 2) for i in range(n)])
slm_centres_y = np.array([eyt + (j * (slm_block_h + g1_y)) + (slm_block_h // 2) for j in range(m)])


######################################################
# DMD #
######################################################


dmd_resX = 1920
dmd_resY = 1080

dmd_w = 1920
dmd_h = 1080

dmd_xc = dmd_resX//2
dmd_yc = dmd_resY//2

dmd_s = np.s_[dmd_xc-dmd_w//2:dmd_xc+dmd_w//2, dmd_yc-dmd_h//2:dmd_yc+dmd_h//2]

dmd_block_w = int(slm_block_w * dmd_w / slm_w)
np.save('./tools/dmd_block_w.npy', np.array([dmd_block_w]))
dmd_block_h = int(slm_block_h * dmd_h / slm_h) - 4

dmd_centres_x = (slm_centres_x * dmd_w / slm_w).astype(int)
dmd_centres_y = (slm_centres_y * dmd_h / slm_h).astype(int)
dmd_centres_x, dmd_centres_y = np.meshgrid(dmd_centres_x, dmd_centres_y)

gpu_dmd_centres_x = cp.asarray(dmd_centres_x, dtype=cp.int32)
gpu_dmd_centres_y = cp.asarray(dmd_centres_y, dtype=cp.int32)

gpu_dmd_centres_x = gpu_dmd_centres_x[cp.newaxis, ...].repeat(dmd_block_w+1, axis=0)

inputs = loadmat('C:/Users/spall/OneDrive - Nexus365/Code/JS/controller/onn_test/MNIST digit - subsampled - 121.mat')

num_train = 60000
num_test = 10000

trainY_raw = inputs['trainY']
trainY = np.zeros((num_train, 10))
for i in range(num_train):
    trainY[i, trainY_raw[0, i]] = 1

testY_raw = inputs['testY']
testY = np.zeros((num_test, 10))
for i in range(num_test):
    testY[i, testY_raw[0, i]] = 1

trainX_raw = inputs['trainX']
trainX = np.empty((num_train, 121))
for i in range(num_train):
    trainX_k = trainX_raw[i, :] - trainX_raw[i, :].min()
    trainX_k = trainX_k / trainX_k.max()
    trainX[i, :] = trainX_k

testX_raw = inputs['testX']
testX = np.empty((num_test, 121))
for i in range(num_test):
    testX_k = testX_raw[i, :] - testX_raw[i, :].min()
    testX_k = testX_k / testX_k.max()
    testX[i, :] = testX_k

np.random.seed(0)
np.random.shuffle(trainX)
np.random.seed(0)
np.random.shuffle(trainY)
np.random.seed(0)
np.random.shuffle(testX)
np.random.seed(0)
np.random.shuffle(testY)

valX = testX[:5000, :].copy()
testX = testX[5000:, :].copy()

valY = testY[:5000, :].copy()
testY = testY[5000:, :].copy()

trainX -= 0.1
trainX = np.clip(trainX, 0, 1)
trainX /= trainX.max()

valX -= 0.1
valX = np.clip(valX, 0, 1)
valX /= valX.max()

testX -= 0.1
testX = np.clip(testX, 0, 1)
testX /= testX.max()

# trainX = 1-trainX
# valX = 1-valX
# testX = 1-testX

trainX = trainX.reshape((60000, 11, 11))[:, 1:, 1:].reshape((60000, 100))
valX = valX.reshape((5000, 11, 11))[:, 1:, 1:].reshape((5000, 100))
testX = testX.reshape((5000, 11, 11))[:, 1:, 1:].reshape((5000, 100))

trainX = (trainX*dmd_block_w).astype(int)/dmd_block_w
valX = (valX*dmd_block_w).astype(int)/dmd_block_w
testX = (testX*dmd_block_w).astype(int)/dmd_block_w


weights = loadmat('./tools/Weights_trained_feb16.mat')['Weight_opt']
# weights1 = np.zeros((126,49))
# weights1[2:-3,:] = weights[0,0]
weights1 = weights[0, 0]
weights1 = weights1 / weights1.max()

weights2 = weights[0, 1].T

# actual_uppers_arr_1024 = np.load('./tools/actual_uppers_arr_1024.npy')
# uppers1_nm = actual_uppers_arr_1024[..., -1].copy()


# def find_shuffle_indxs(weight_arr):
#
#     # empty arrays for storing column shuffles
#     mapped_cols_indxs_sorted_arr = np.empty((n, m))
#
#     # find indices that sort rows weakest to brightest
#     weights1_rows_indxs = np.argsort(np.abs(weight_arr).max(axis=0)).astype(int)
#     uppers1_rows_indxs = np.argsort(uppers1_nm.max(axis=0)).astype(int)
#
#     # reorder weights so that now, value V in index k means:
#     # 'row V should go to row k'
#     mapped_rows_indxs_sorted = uppers1_rows_indxs[np.argsort(weights1_rows_indxs)]
#
#     for k in range(m):
#         # select the kth weight row
#         weights1_row = weight_arr[:, k]
#
#         # and the row it will be sent to
#         mapped_row_indx = mapped_rows_indxs_sorted[k]
#         uppers1_row = uppers1_nm[:, mapped_row_indx]
#
#         # find the indices that sort columns in this row
#         weights1_col_indxs = np.argsort(np.abs(weights1_row))
#         uppers1_col_indxs = np.argsort(np.abs(uppers1_row))
#
#         mapped_cols_indxs_sorted = uppers1_col_indxs[np.argsort(weights1_col_indxs)]
#
#         mapped_cols_indxs_sorted_arr[:, k] = mapped_cols_indxs_sorted
#
#     return mapped_rows_indxs_sorted, mapped_cols_indxs_sorted_arr
#
#
# mapped_rows_global_indxs_sorted, mapped_cols1_indxs_sorted_arr = find_shuffle_indxs(weights1)
#
#
# def map_weights1(weight_arr):
#     mapped = np.empty((n, m))
#
#     for l in range(n):
#         for k in range(m):
#             val = weight_arr[l, k]
#             mapped_row_indx = int(mapped_rows_global_indxs_sorted[k])
#             mapped_col_indx = int(mapped_cols1_indxs_sorted_arr[l, k])
#             mapped[mapped_col_indx, mapped_row_indx] = val
#
#     return mapped
#
#
# weights1_mapped_scaled = map_weights1(weights1) * 0.8
# np.save('./MNIST/weights1_mapped_scaled.npy', weights1_mapped_scaled)
#
#
r = cp.array([int((-1) ** i * cp.ceil(i / 2)) for i in range(dmd_block_w+1)])
ws, ms, ns = cp.meshgrid(cp.arange(dmd_block_w+1), cp.arange(m), cp.arange(n), indexing='ij')
dmd_xc_shifted = gpu_dmd_centres_x - r[:, np.newaxis, cp.newaxis]
dmd_yc = gpu_dmd_centres_y[cp.newaxis, ...].repeat(dmd_block_w+1, axis=0)
bits24 = 2**cp.arange(8, dtype=cp.uint8)
bits24 = bits24[:, cp.newaxis, cp.newaxis]
# ms2d, ns2d = cp.meshgrid(cp.arange(m), cp.arange(n))
# gpu_mapped_rows_indxs_sorted = cp.array(mapped_rows_global_indxs_sorted)
# gpu_mapped_cols1_indxs_sorted_arr = cp.array(mapped_cols1_indxs_sorted_arr)


def map_vec_to_arr(vector, shuffle=True, ref=False):

    global ref_col

    vector = (cp.array(vector) * dmd_block_w).astype(cp.uint8)

    if vector.shape == (n,):
        vals = vector[:, cp.newaxis].repeat(m, axis=1)
    else:
        vals = vector.copy()

    if ref:
        # vals[:, 0] = dmd_block_w
        # vals[:, -1] = dmd_block_w
        vals[ref_col, m//2] = dmd_block_w

        vals[:, m // 2] = dmd_block_w

    # if shuffle:
    #     mapped_row_indxs = gpu_mapped_rows_indxs_sorted[cp.arange(m)]
    #     mapped_row_indxs = mapped_row_indxs[cp.newaxis, :].repeat(n, axis=0)
    #     mapped_col_indxs = gpu_mapped_cols1_indxs_sorted_arr[ns2d, ms2d].astype(cp.int)
    #     mapped = cp.zeros((n, m), dtype=cp.uint8)
    #     mapped[mapped_col_indxs, mapped_row_indxs] = vals
    #     return mapped
    #
    # else:

    # vals[:, 1::2] = 0

    return vals


def make_dmd_image(vector, shuffle=True, ref=False):

    mapped = map_vec_to_arr(vector, shuffle, ref)

    target = mapped[..., cp.newaxis].repeat(dmd_block_w + 1, axis=-1)

    mask = (target > cp.arange(dmd_block_w + 1))
    mask = mask.transpose(2, 1, 0)

    mapped = cp.zeros((dmd_block_w + 1, dmd_h, dmd_w), dtype='bool')

    for j in range(dmd_block_h // 2):
        mapped[ws, dmd_yc - j, dmd_xc_shifted] = mask
        mapped[ws, dmd_yc + j + 1, dmd_xc_shifted] = mask

    out = mapped.sum(axis=0)

    # out = cp.flip(out, 0).astype(cp.uint8)
    out = cp.flip(out, 1).astype(cp.uint8)

    return out


def make_dmd_rgb(targets_list, shuffle=True, ref=False):
    assert len(targets_list) % 24 == 0

    dmd_imgs = []

    for k in range(len(targets_list) // 24):

        targets = targets_list[k * 24:(k + 1) * 24]

        dmd_r = make_dmd_image(targets[0], shuffle, ref)
        for i in range(1, 8):
            dmd_r += make_dmd_image(targets[i], shuffle, ref) * (2 ** (i % 8))

        dmd_g = make_dmd_image(targets[8], shuffle, ref)
        for i in range(9, 16):
            dmd_g += make_dmd_image(targets[i], shuffle, ref) * (2 ** (i % 8))

        dmd_b = make_dmd_image(targets[16], shuffle, ref)
        for i in range(17, 24):
            dmd_b += make_dmd_image(targets[i], shuffle, ref) * (2 ** (i % 8))

        dmd_rgb = cp.stack((dmd_r, dmd_g, dmd_b,
                            cp.full((dmd_h, dmd_w), 255, dtype=cp.uint8)), axis=-1).astype(cp.uint8)

        dmd_imgs.append(dmd_rgb)

        del dmd_r, dmd_b, dmd_g, dmd_rgb

    return dmd_imgs


if __name__ == '__main__':

    # test_arrs = []
    #
    # for i in np.linspace(0, 0.8, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0.4, 0.8, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0., 0.4, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.flip(np.linspace(0.4, 0.8, 24)):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0., 0.4, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.flip(np.linspace(0, 0.8, 24)):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0, 0.8, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0., 0.4, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0, 0.8, 24):
    #     test_arrs.append(np.full(n, i))
    #
    # for i in np.linspace(0.4, 0.8, 24):
    #     test_arrs.append(np.full(n, i))

    # test_arrs1 = []
    # for i in range(5):
    #     test_arrs1.extend(test_arrs)
    #
    # test_arrs = test_arrs1
    # print(len(test_arrs))

    # dmd_vecs = []
    # dmd_arrs = []
    #
    # for vec in test_arrs:
    #     dmd_vec = (vec * dmd_block_w).astype(np.uint32) / dmd_block_w
    #     dmd_arr = map_vec_to_arr(vec) / dmd_block_w
    #
    #     # print(dmd_vec.shape, dmd_vec.mean(), dmd_vec.max())
    #     # print(dmd_arr.shape, dmd_arr.mean(), dmd_arr.max())
    #
    #     dmd_vecs.append(dmd_vec)
    #     dmd_arrs.append(dmd_arr)
    #
    # cp.save('./MNIST/test_frames/targets/target_vecs.npy', cp.array(dmd_vecs))
    # cp.save('./MNIST/test_frames/targets/target_arrs.npy', cp.array(dmd_arrs))
    #
    # target_frames = make_dmd_rgb(test_arrs)
    # print(len(target_frames))
    #
    # cp.save('./MNIST/test_frames/frames/frames.npy', cp.array(target_frames))


    # batch_size = 24
    #
    # for k in range(2500):
    #
    #     input_arrs = []
    #     for i in range(k * batch_size, (k + 1) * batch_size):
    #         vec = trainX[i, :].copy()
    #         arr = vec[:, cp.newaxis].repeat(m, axis=1)
    #         arr = np.insert(arr, 50, np.ones((21, arr.shape[1])), 0)
    #         arr[:, m//2] = 1.
    #         input_arrs.append(arr)
    #
    #     batch_k = make_dmd_rgb(input_arrs, shuffle=False, ref=False)
    #
    #     xs_k = [(input_arrs[i] * dmd_block_w).astype(np.uint32) / dmd_block_w for i in range(batch_size)]
    #     ys_k = [trainY[i, :] for i in range(k * batch_size, (k + 1) * batch_size)]
    #
    #     cp.save('./MNIST/trainX_rgb_frames_m24_no_invert/rgb24_{}'.format(k), cp.array(batch_k))
    #     cp.save('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_M24/xs/xs_{}'.format(k), cp.array(xs_k))
    #     cp.save('D:/MNIST/trainX/SHUFFLE_0_NOREMAP_M24/ys/ys_{}'.format(k), cp.array(ys_k))
    #
    #     if k%10==0:
    #         sys.stdout.write("\rsaved batch {}".format(k))
    #         sys.stdout.flush()
    #
    #     del batch_k, input_arrs
    #     cp._default_memory_pool.free_all_blocks()


    batch_size = 240
    xs = np.full((4800, n, m), np.nan)
    ys = np.full((4800, 10), np.nan)

    for k in range(0, 20):

        input_arrs = []
        for i in range(k * batch_size, (k + 1) * batch_size):
            vec = valX[i, :].copy()
            arr = vec[:, cp.newaxis].repeat(m, axis=1)
            arr = np.insert(arr, 50, np.ones((21, arr.shape[1])), 0)
            arr[:, m//2] = 1.
            input_arrs.append(arr)

        batch_k = make_dmd_rgb(input_arrs, shuffle=False, ref=False)

        xs_k = [(input_arrs[i] * dmd_block_w).astype(np.uint32) / dmd_block_w for i in range(batch_size)]
        ys_k = [valY[i, :] for i in range(k * batch_size, (k + 1) * batch_size)]

        xs[k * 240:(k + 1) * 240, ...] = xs_k
        ys[k * 240:(k + 1) * 240, :] = ys_k

        cp.save('D:/MNIST/valX/SHUFFLE_0_NOREMAP_M24/frames/rgb24_{}'.format(k), cp.array(batch_k))

        sys.stdout.write("\rsaved batch {}".format(k))
        sys.stdout.flush()

        del batch_k, input_arrs
        cp._default_memory_pool.free_all_blocks()

    cp.save('D:/MNIST/valX/SHUFFLE_0_NOREMAP_M24/xs.npy', cp.array(xs))
    cp.save('D:/MNIST/valX/SHUFFLE_0_NOREMAP_M24/ys.npy', cp.array(ys))


    # batch_size = 240
    # xs = np.full((4800, n), np.nan)
    # ys = np.full((4800, 10), np.nan)
    #
    # for k in range(0, 20):
    #
    #     input_arrs = [testX[i, :] for i in range(k * batch_size, (k + 1) * batch_size)]
    #
    #     batch_k = make_dmd_rgb(input_arrs, shuffle=False)
    #
    #     xs_k = [(input_arrs[i] * dmd_block_w).astype(np.uint32) / dmd_block_w for i in range(batch_size)]
    #     ys_k = [testY[i, :] for i in range(k * batch_size, (k + 1) * batch_size)]
    #
    #     xs[k * 240:(k + 1) * 240, :] = xs_k
    #     ys[k * 240:(k + 1) * 240, :] = ys_k
    #
    #     cp.save('./MNIST/testX/SHUFFLE_0_NOREMAP_INDIVIDUAL/frames/rgb24_{}'.format(k), cp.array(batch_k))
    #
    #     sys.stdout.write("\rsaved batch {}".format(k))
    #     sys.stdout.flush()
    #
    #     del batch_k, input_arrs
    #     cp._default_memory_pool.free_all_blocks()
    #
    # cp.save('./MNIST/testX/SHUFFLE_0_NOREMAP_INDIVIDUAL/xs.npy', cp.array(xs))
    # cp.save('./MNIST/testX/SHUFFLE_0_NOREMAP_INDIVIDUAL/ys.npy', cp.array(ys))

    # ref_col = 62

    # for i in range(5):
    #
    #     print(i)
    #
    #     vecs = []
    #
    #     for j in range(i*24, (i+1)*24):
    #         col_vec = np.zeros(n)
    #         col_vec[j] = 1
    #
    #         vecs.append(col_vec)
    #
    #     rgb = make_dmd_rgb(vecs, shuffle=False, ref=True)
    #
    #     cp.save('./tools/dmd_imgs/cols/col_array_{}'.format(i), cp.array(rgb))
    #
    #     del vecs, rgb


    # for i in range(5):
    #
    #     print(i)
    #
    #     vecs = []
    #
    #     for j in range(i*24, (i+1)*24):
    #         col_vec = np.zeros((n, m))
    #         col_vec[ref_col, :] = 1
    #
    #         col_vec[j, :] = 1
    #         col_vec[j, m//2] = 0
    #
    #         # col_vec[ref_col, m // 2] = 0.
    #         col_vec[ref_col + 40, m // 2] = dmd_block_w
    #         col_vec[ref_col - 40, m // 2] = dmd_block_w
    #
    #         vecs.append(col_vec)
    #
    #     rgb = make_dmd_rgb(vecs, shuffle=False, ref=False)
    #
    #     cp.save('./tools/dmd_imgs/two_cols/col_array_{}'.format(i), cp.array(rgb))
    #
    #     del vecs, rgb

