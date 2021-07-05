import numpy as np
import cupy as cp
import sys

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])

######################################################
# SLM1 (Santec) #
######################################################

slm_resX = 1920
slm_resY = 1080

slm_resX_actual = 1440
slm_resY_actual = 1050

slm_xc = slm_resX_actual // 2
slm_yc = slm_resY_actual // 2

slm_w = 1406
slm_h = 774

slm_s = np.s_[slm_xc - (slm_w // 2):slm_xc + (slm_w // 2), slm_yc - (slm_h // 2):slm_yc + (slm_h // 2)]

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

# create index arrays
x_SLM = slm_resX_actual//2
y_SLM = slm_resY_actual//2
X_array = np.arange(-x_SLM, x_SLM)
Y_array = np.arange(-y_SLM, y_SLM)
Y, X = np.meshgrid(Y_array, X_array)

# create the phase to GL LUT
GL_lut_arr = np.load('./tools/Santec_LUT_NEW_4PI.npy')
max_GL = GL_lut_arr.shape[0]
d_phase = GL_lut_arr[-1, 0]/max_GL
gpu_LUT_arr = cp.asarray(GL_lut_arr, dtype='float32')

slm_wf = np.load('./tools/slm_wf.npy')[:, :-1]
phase_offset = np.load('./tools/phase_offset.npy')[:, :-1]
gpu_slm_wf = cp.asarray(slm_wf-phase_offset, dtype='float32')

period = 10
gr = 2*np.pi*(-X+Y)/period+1e-5
gpu_gr = cp.asarray(gr, dtype='float32')

step = 0.00001
inv_sinc_LUT = np.load("./tools/inv_sinc_lut.npy")
gpu_inv_sinc_LUT = cp.asarray(inv_sinc_LUT, dtype='float32')


def make_slm_image(gpu_target_in):

    slm_image = cp.repeat(gpu_target_in, slm_block_w + g1_x, axis=0)
    slm_image = cp.repeat(slm_image, slm_block_h + g1_y, axis=1)

    for i in range(g1_x):
        slm_image[slm_block_w + i::slm_block_w + g1_x, :] = 0
    for j in range(g1_y):
        slm_image[:, slm_block_h + j::slm_block_h + g1_y] = 0

    slm_image_aoi = cp.zeros((slm_w, slm_h), dtype='float16')
    slm_image = slm_image[:slm_w - g1_x, :slm_h - g1_y]

    slm_image = np.flip(slm_image, axis=1)

    slm_image_aoi[exl:slm_w - exr, eyt:slm_h - eyb] = slm_image

    del slm_image

    return slm_image_aoi


# max_offset_phi = 7 * np.pi
# phase_offset_arr_nm = cp.zeros((n, m))
#
# for i in range(m):
#
#     if i % 2 == 0:
#         lin_phase = cp.linspace(0, max_offset_phi / 2, n)
#     else:
#         lin_phase = np.flip(cp.linspace(0, max_offset_phi / 2, n))
#
#     lin_phase = np.mod(lin_phase, 2 * np.pi)
#     phase_offset_arr_nm[:, i] = lin_phase
#
# phase_offset_arr_aoi = make_slm_image(phase_offset_arr_nm)
# phase_offset_arr = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
# phase_offset_arr[slm_s] = phase_offset_arr_aoi

def make_slm_rgb(gpu_target_A, gpu_target_phi=None, stagger=0):

    gpu_A_aoi = make_slm_image(cp.abs(gpu_target_A))

    if gpu_target_phi is not None:
        gpu_phi_aoi = make_slm_image(gpu_target_phi) + gpu_slm_wf
    else:
        gpu_phi_aoi = make_slm_image(cp.angle(gpu_target_A)) + gpu_slm_wf

    gpu_A = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
    gpu_A[slm_s] = gpu_A_aoi

    gpu_phi = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
    gpu_phi[slm_s] = gpu_phi_aoi

    gpu_A = cp.maximum(gpu_A, 1e-16, dtype='float32')
    gpu_A = cp.minimum(gpu_A, 1 - 1e-4, dtype='float32')

    gpu_A *= 0.99

    gpu_M = 1 - gpu_inv_sinc_LUT[(gpu_A / step).astype(cp.int)]
    gpu_F = gpu_phi + cp.pi * (1 - gpu_M)
    gpu_o = gpu_M * cp.mod(gpu_F + gpu_gr, 2 * cp.pi)
    # gpu_o += cp.pi

    # if stagger:
    #     gpu_o += phase_offset_arr

    # phase-GL LUT
    gpu_idx_phase = (gpu_o / d_phase).astype(int)
    gpu_idx_phase = cp.clip(gpu_idx_phase, 0, max_GL - 1).astype(int)
    gpu_o_GL = gpu_LUT_arr[gpu_idx_phase, 1]

    gpu_out = cp.zeros((slm_resX, slm_resY, 1), dtype='float16')
    gpu_out[:slm_resX_actual, :slm_resY_actual, 0] = gpu_o_GL
    gpu_out = gpu_out.transpose(1, 0, 2)

    # 10-bit santec encoding
    gpu_r_array = gpu_out // 128
    gpu_g_array = gpu_out // 16 - gpu_r_array * 8
    gpu_b_array = gpu_out - (gpu_out // 16) * 16

    gpu_r_array = gpu_r_array * 32
    gpu_g_array = gpu_g_array * 32
    gpu_b_array = gpu_b_array * 16

    gpu_r_array = gpu_r_array.astype(cp.uint8)
    gpu_g_array = gpu_g_array.astype(cp.uint8)
    gpu_b_array = gpu_b_array.astype(cp.uint8)

    gpu_color_array = cp.concatenate((gpu_r_array, gpu_g_array, gpu_b_array), axis=2)

    gpu_color_array.astype(cp.uint8)

    del gpu_r_array, gpu_g_array, gpu_b_array, gpu_out
    del gpu_idx_phase, gpu_o, gpu_M, gpu_A, gpu_F, gpu_A_aoi, gpu_phi_aoi, gpu_phi

    cp._default_memory_pool.free_all_blocks()

    return gpu_color_array


def slm_rm_case_10(seed):
    seed += 6

    rm = np.empty((n, m))

    for j in range(m):
        row = 3 * j

        np.random.seed(row + 100 * seed + 1)

        lower = -1
        upper = 1
        mean = (upper - lower) * np.random.random() + lower

        np.random.seed(row + 100 * seed + 2)

        lower = 0.2
        upper = 0.4
        std = (upper - lower) * np.random.random() + lower

        np.random.seed(row + 100 * seed + 3)
        rm[:, j] = np.random.normal(loc=mean, scale=std, size=n)

    rm = np.clip(rm, -1, 1)

    return rm


if __name__ == '__main__':

    arr = np.random.rand(n, m)
    slm1_arr = make_slm_rgb(arr)

    print(slm1_arr.shape)
