import numpy as np
import cupy as cp
import sys

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])

ref_spot = m//2

######################################################
# SLM1 (Santec) #
######################################################

######## santec ########

slm_resX = 1920
slm_resY = 1080

slm_resX_actual = 1440
slm_resY_actual = 1050

slm_xc = slm_resX_actual//2
slm_yc = slm_resY_actual//2

slm_w = 1406
slm_h = 775

slm_ref_w = 300
slm_sig_w = slm_w - slm_ref_w

slm_s = np.s_[slm_xc-(slm_w//2):slm_xc+slm_w-(slm_w//2), slm_yc-(slm_h//2):slm_yc+slm_h-(slm_h//2)]

dmd_resX = 1920
dmd_resY = 1080

dmd_w = 1920
dmd_h = 1080

# slm_block_w = int(((slm_w + 1) / n))
# slm_block_h = int(((slm_h + 1) / m))
#
# g1_x = (slm_w - (slm_block_w * n)) // (n - 1)
# g1_y = (slm_h - (slm_block_h * m)) // (m - 1)
#
# exl = (slm_w - (slm_block_w * n) - (n - 1) * g1_x) // 2
# exr = (slm_w - (slm_block_w * n) - (n - 1) * g1_x) - exl
# eyt = (slm_h - (slm_block_h * m) - (m - 1) * g1_y) // 2
# eyb = (slm_h - (slm_block_h * m) - (m - 1) * g1_y) - eyt + 1
#
# slm_centres_x = np.array([exl + (i * (slm_block_w + g1_x)) + (slm_block_w // 2) for i in range(n)])
# slm_centres_y = np.array([eyt + (j * (slm_block_h + g1_y)) + (slm_block_h // 2) for j in range(m)])

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

slm_wf = np.load('./tools/slm_wf.npy')
gpu_slm_wf = cp.asarray(slm_wf, dtype='float32')

period = 10
gr = 2*np.pi*(-X+Y)/period+1e-5
gpu_gr = cp.asarray(gr, dtype='float32')

step = 0.00001
inv_sinc_LUT = np.load("./tools/inv_sinc_lut.npy")
gpu_inv_sinc_LUT = cp.asarray(inv_sinc_LUT, dtype='float32')


def update_params(ref_block_val, batch_size, num_frames):

    global slm_w, slm_sig_w, slm_ref_w, slm_h, ref_spot
    global cols_to_del, rows_to_del, slm_block_w, slm_block_h
    global dmd_block_w, dmd_block_h, slm_centers_x, slm_centers_y, insert_indx, insert_indx_dmd
    global gpu_dmd_centers_x, gpu_dmd_centers_y
    global slm_block_w, slm_block_h
    global ws, dmd_xc_shifted, r
    global y_blocks, y_blocks_multi, y_blocks_multi_cp
    global ampl_ref_sindx, ampl_ref_eindx, ampl_ref_s, bit_values, bit_values_extended
    global o, u, e

    if ref_block_val is not None:
        slm_w_local = slm_sig_w
    else:
        slm_w_local = slm_w

    slm_block_w = int(slm_w_local / n) + 1
    slm_block_h = int(slm_h / m) + 1

    cols_to_del = np.linspace(0, slm_block_w * n - 1, (slm_block_w * n) - slm_w_local).astype(np.int)
    rows_to_del = np.linspace(0, slm_block_h * m - 1, (slm_block_h * m) - slm_h).astype(np.int)

    # some magic to figure out where to place ref block in between blocks, so it doesnt split a block in half
    insert_indx = slm_block_w * (n // 2) - (cols_to_del < slm_block_w * (n // 2)).sum()

    if ref_block_val is None:
        slm_edges_x = np.linspace(0, slm_w, n + 1)
        slm_centers_x = np.array([(slm_edges_x[i] + slm_edges_x[i + 1]) / 2 for i in range(n)]).astype(int)

    else:
        slm_edges_x1 = np.linspace(0, insert_indx, (n // 2) + 1).astype(int)
        slm_centers_x1 = np.array([(slm_edges_x1[i] + slm_edges_x1[i + 1]) / 2 for i in range(n // 2)]).astype(int)

        slm_edges_x2 = np.linspace(insert_indx + slm_ref_w, slm_w, n - (n // 2) + 1).astype(int)
        slm_centers_x2 = np.array([(slm_edges_x2[i] + slm_edges_x2[i + 1]) / 2 for i in range(n - n // 2)]).astype(int)

        slm_centers_x = np.hstack((slm_centers_x1, slm_centers_x2))

    slm_edges_y = np.linspace(0, slm_h, m + 1)
    slm_centers_y = np.array([(slm_edges_y[i] + slm_edges_y[i + 1]) / 2 for i in range(m)]).astype(int)

    # slm_centers_x_grid, slm_centers_y_grid = np.meshgrid(slm_centers_x, slm_centers_y)

    dmd_block_w = int((slm_block_w - 1) * dmd_w / slm_w)
    dmd_block_h = int((slm_block_h - 1) * 0.5 * dmd_h / slm_h)

    if dmd_block_h % 2 == 0:
        dmd_block_h -= 1

    dmd_centers_x = (slm_centers_x * dmd_w / slm_w).astype(int)
    dmd_centers_y = (slm_centers_y * dmd_h / slm_h).astype(int)
    dmd_centers_x_grid, dmd_centers_y_grid = np.meshgrid(dmd_centers_x, dmd_centers_y)

    insert_indx_dmd = int(insert_indx * dmd_w / slm_w)

    gpu_dmd_centers_x = cp.array(dmd_centers_x_grid)
    gpu_dmd_centers_y = cp.array(dmd_centers_y_grid)

    r = cp.array([int((-1) ** i * cp.ceil(i / 2)) for i in range(dmd_block_w + 1)])
    ws, ns = cp.meshgrid(cp.arange(dmd_block_w + 1), cp.arange(n), indexing='ij')
    dmd_xc_shifted = gpu_dmd_centers_x[0, :] - r[:, cp.newaxis]

    y_blocks = cp.zeros(dmd_h, dtype='bool')
    for j in range(dmd_block_h // 2 + 1):
        y_blocks[gpu_dmd_centers_y[:, 0] - j] = 1
        y_blocks[gpu_dmd_centers_y[:, 0] + j] = 1

    y_blocks_multi = y_blocks[cp.newaxis, :].get().astype(np.bool)
    y_blocks_multi_cp = cp.array(y_blocks_multi)

    ampl_ref_sindx = int(gpu_dmd_centers_y[ref_spot, 0] - (dmd_block_h // 2) + 1)
    ampl_ref_eindx = int(gpu_dmd_centers_y[ref_spot, 0] + (dmd_block_h // 2))

    ampl_ref_s = np.s_[:, :, ampl_ref_sindx:ampl_ref_eindx]

    bit_values = cp.zeros((3, 24))
    bit_values[0, :8] = cp.array([2**ii for ii in range(8)])
    bit_values[1, 8:16] = cp.array([2**ii for ii in range(8)])
    bit_values[2, 16:] = cp.array([2**ii for ii in range(8)])

    bit_values_extended = cp.repeat(bit_values[:, None, :, None, None], num_frames, axis=1).astype(cp.uint8)

    o = dmd_w
    u = dmd_h
    e = batch_size

    return dmd_block_w


# def make_slm_image(gpu_target_in, ref_block_val):
#
#     global slm_w, slm_sig_w, slm_ref_w, slm_h
#     global cols_to_del, rows_to_del, slm_block_w, slm_block_h
#
#     slm_image = np.repeat(gpu_target_in.get(), slm_block_w, axis=0)
#     slm_image = np.repeat(slm_image, slm_block_h, axis=1)
#     slm_image = np.delete(slm_image, cols_to_del, 0)
#     slm_image = np.delete(slm_image, rows_to_del, 1)
#
#     if ref_block_val is not None:
#         slm_image = np.insert(slm_image, insert_indx, np.ones((slm_ref_w, slm_image.shape[1])) * ref_block_val, 0)
#
#     return slm_image


def make_slm_rgb(target, ref_block_val, ref_block_phase=0.):

    global slm_w, slm_sig_w, slm_ref_w, slm_h
    global cols_to_del, rows_to_del, slm_block_w, slm_block_h

    target = np.flip(target, axis=1)

    aoi = np.repeat(target, slm_block_w, axis=0)
    aoi = np.repeat(aoi, slm_block_h, axis=1)
    aoi = np.delete(aoi, cols_to_del, 0)
    aoi = np.delete(aoi, rows_to_del, 1)

    A_aoi = np.abs(aoi)
    phi_aoi = np.angle(aoi)

    if ref_block_val is not None:
        A_aoi = np.insert(A_aoi, insert_indx, np.ones((slm_ref_w, A_aoi.shape[1])) * ref_block_val, 0)
        phi_aoi = np.insert(phi_aoi, insert_indx, np.ones((slm_ref_w, phi_aoi.shape[1])) * ref_block_phase, 0)

    gpu_A_aoi = cp.array(A_aoi)
    gpu_phi_aoi = cp.array(phi_aoi) + gpu_slm_wf

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


def make_dmd_image(arr, ref=1, ref_block_val=1.):

    global r

    arr = cp.array(arr*dmd_block_w)

    ws_local, ms_local, ns_local = cp.meshgrid(cp.arange(dmd_block_w + 1), cp.arange(m), cp.arange(n),
                                               indexing='ij')
    dmd_xc = gpu_dmd_centers_x - r[:, cp.newaxis, cp.newaxis]
    dmd_yc = gpu_dmd_centers_y[cp.newaxis, ...].repeat(dmd_block_w + 1, axis=0)

    target = arr[..., cp.newaxis].repeat(dmd_block_w + 1, axis=-1)

    mask = (target > cp.arange(dmd_block_w + 1))
    mask = mask.transpose(2, 1, 0)

    mapped = cp.zeros((dmd_block_w + 1, dmd_h, dmd_w), dtype='bool')

    for j in range(dmd_block_h // 2):
        mapped[ws_local, dmd_yc - j, dmd_xc] = mask
        mapped[ws_local, dmd_yc + j + 1, dmd_xc] = mask

    out = mapped.sum(axis=0).astype(cp.uint8)

    if ref:
        out[ampl_ref_sindx:ampl_ref_eindx, :] = 1

    if ref_block_val is not None:
        if ref_block_val > 0:
            out[:, insert_indx_dmd:insert_indx_dmd + int(slm_ref_w * dmd_w / slm_w)] = 1
        else:
            out[:, insert_indx_dmd:insert_indx_dmd + int(slm_ref_w * dmd_w / slm_w)] = 0

    out = cp.flip(out, 0)

    rgb = out[..., None].repeat(4, axis=2)
    rgb *= 255

    rgb[..., -1] = 255

    return rgb.astype(cp.uint8)


def make_dmd_batch(vecs, ref, ref_block_val, batch_size, num_frames):

    def find_1d_pattern(vec):
        target = vec[cp.newaxis, :].astype(cp.uint8).repeat(dmd_block_w + 1, axis=0)
        mask = (target > cp.arange(dmd_block_w + 1)[:, cp.newaxis])
        mapped = cp.zeros((dmd_block_w + 1, dmd_w), dtype='bool')
        mapped[ws, dmd_xc_shifted] = mask
        out = mapped.sum(axis=0).astype(cp.bool)
        return cp.flip(out)

    outxs = cp.empty((batch_size, dmd_w), dtype='bool')
    for indx, vec in enumerate(vecs):
        outxs[indx, :] = find_1d_pattern(vec*dmd_block_w)

    imgs = cp.einsum('eo,eu->eou', outxs, y_blocks_multi_cp).astype(cp.bool)

    if ref:
        imgs[ampl_ref_s] = 1

    if ref_block_val is not None:
        if ref_block_val > 0:
            imgs[:, insert_indx_dmd:insert_indx_dmd + int(slm_ref_w * dmd_w / slm_w)] = 1
        else:
            imgs[:, insert_indx_dmd:insert_indx_dmd + int(slm_ref_w * dmd_w / slm_w)] = 0

    imgs = imgs.reshape((num_frames, 24, dmd_w, dmd_h)).astype(cp.uint8)[None, ...].repeat(3, axis=0)
    imgs *= bit_values_extended
    imgs = imgs.sum(axis=2)
    imgs = cp.transpose(imgs, (1, 3, 2, 0))
    imgs = cp.flip(imgs, axis=1)
    del outxs

    return imgs


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
