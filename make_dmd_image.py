import numpy as np
import cupy as cp
import sys

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
dmd_block_h = int(slm_block_h * dmd_h / slm_h) - 4

dmd_centres_x = (slm_centres_x * dmd_w / slm_w).astype(int)
dmd_centres_y = (slm_centres_y * dmd_h / slm_h).astype(int)
dmd_centres_x, dmd_centres_y = np.meshgrid(dmd_centres_x, dmd_centres_y)

gpu_dmd_centres_x = cp.asarray(dmd_centres_x, dtype=cp.int32)
gpu_dmd_centres_y = cp.asarray(dmd_centres_y, dtype=cp.int32)


def make_dmd_image(arr_in):
    """take a target array in [0,1] size (n1,m1)
    and make DMD image size (dmd_resX, dmd_resY) with varying widths"""

    # convert target ampl to target widths
    target = (arr_in * dmd_block_w).astype(cp.uint32)

    dmd_img = cp.zeros((dmd_w, dmd_h), dtype=cp.int32)

    for i in range(dmd_block_w):
        mask = (target > i).T.astype(cp.int)
        dmd_temp = cp.zeros((dmd_w, dmd_h), dtype=cp.int32)
        dmd_temp[gpu_dmd_centres_x, gpu_dmd_centres_y] = mask
        roll_by = cp.int((-1) ** i * np.ceil(i / 2))
        shifted = cp.roll(dmd_temp, roll_by, axis=0)
        dmd_img += shifted

    dmd_img_1 = dmd_img.copy()

    for j in range(1, (dmd_block_h // 2) + 1):
        shifted = cp.roll(dmd_img, j, axis=1)
        dmd_img_1 += shifted

        shifted = cp.roll(dmd_img, -j, axis=1)
        dmd_img_1 += shifted

    dmd_out = cp.zeros((dmd_resX, dmd_resY), dtype=cp.uint8)
    dmd_out[dmd_s] = dmd_img_1

    dmd_out = dmd_out.T.astype(cp.uint8)

    # dmd_out = cp.flip(dmd_out, 1).astype(cp.uint8)
    # dmd_out = cp.flip(dmd_out, 0).astype(cp.uint8)

    return dmd_out


def make_dmd_rgb(samples):
    dmd_24_arr = cp.zeros((dmd_resY, dmd_resX, 8, 4), dtype=cp.uint8)

    for i in range(8):
        dmd_24_arr[..., i, 0] = samples[i] * (2 ** (i % 8))

    for i in range(8, 16):
        dmd_24_arr[..., i % 8, 1] = samples[i] * (2 ** (i % 8))

    for i in range(16, 24):
        dmd_24_arr[..., i % 8, 2] = samples[i] * (2 ** (i % 8))

    dmd_24_arr = cp.sum(dmd_24_arr, axis=2)
    dmd_24_arr[..., -1] = 255

    return dmd_24_arr.astype(cp.uint8)


def make_dmd_rgb_multi(samples):

    rgbs = []

    num = len(samples)
    assert num % 24 == 0
    for i in range(num // 24):
        samples24 = samples[i * 24: (i + 1) * 24]
        rgb = make_dmd_rgb(samples24)
        rgbs.append(rgb)

    return rgbs




if __name__ == '__main__':
    arr_list = [make_dmd_image(cp.random.rand(n, m)) for _ in range(24)]
    dmd_arr = make_dmd_rgb(arr_list)



