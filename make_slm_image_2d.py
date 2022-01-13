import numpy as np
import cupy as cp

##########
# SANTEC #
##########

slm_resX = 1920
slm_resY = 1080

slm_resX_actual = 1440
slm_resY_actual = 1050

slm_xc = slm_resX_actual//2
slm_yc = slm_resY_actual//2

slm_w = 1406
slm_h = 775

slm_x0 = slm_xc-(slm_w//2)
slm_y0 = slm_yc-(slm_h//2)

slm_R1_w = 300
slm_sig_w = slm_w - slm_R1_w
slm_R2_h = 150
slm_label_h = 100
slm_H1_h = 0  # 50
slm_sig_h = slm_h - slm_R2_h - slm_label_h - slm_H1_h

slm_s = np.s_[slm_x0:slm_x0+slm_w,
              slm_y0:slm_y0+slm_h]
slm_sig_s = np.s_[slm_x0:slm_x0+slm_sig_w,
                  slm_y0:slm_y0+slm_sig_h]
slm_label_s = np.s_[slm_x0:slm_x0+slm_sig_w,
                    slm_y0+slm_sig_h:slm_y0+slm_sig_h+slm_label_h]
slm_R1_s = np.s_[slm_x0+slm_sig_w:slm_x0+slm_w,
                 slm_y0:slm_y0+slm_h]
slm_R2_s = np.s_[slm_x0:slm_x0+slm_sig_w,
                 slm_y0+slm_sig_h+slm_label_h:slm_y0+slm_sig_h+slm_label_h+slm_R2_h]
slm_H1_s = np.s_[slm_x0:slm_x0+slm_sig_w,
                 slm_y0+slm_sig_h+slm_R2_h+slm_label_h:slm_y0+slm_sig_h+slm_R2_h+slm_label_h+slm_H1_h]
slm_V1_s = np.s_[slm_x0+slm_sig_w:slm_x0+slm_w,
                 slm_y0+slm_sig_h+slm_R2_h+slm_label_h:slm_y0+slm_sig_h+slm_R2_h+slm_label_h+slm_H1_h]

#######
# DMD #
#######

dmd_resX = 1920
dmd_resY = 1080

dmd_w = 1920
dmd_h = 1080

dmd_xc = dmd_resX//2
dmd_yc = dmd_resY//2

dmd_x0 = dmd_xc-(dmd_w//2)
dmd_y0 = dmd_yc-(dmd_h//2)

dmd_sig_w = int(slm_sig_w * dmd_w/slm_w)
dmd_sig_h = int(slm_sig_h * dmd_h/slm_h)

dmd_R1_w = int(0.85 * slm_R1_w * dmd_w/slm_w)
dmd_R2_h = int(0.85 * slm_R2_h * dmd_h/slm_h)
dmd_label_h = int(0.85 * slm_label_h * dmd_h/slm_h)

dmd_R1_gap = ((dmd_w - dmd_sig_w) - dmd_R1_w)//2
dmd_R2_gap = ((dmd_h - dmd_sig_h) - dmd_R2_h - dmd_label_h)//3

dmd_H1_h = int(slm_H1_h * dmd_h/slm_h)

label_block_w = dmd_sig_w//10

dmd_s = np.s_[dmd_x0:dmd_x0+dmd_w,
              dmd_y0:dmd_y0+dmd_h]
dmd_sig_s = np.s_[dmd_x0:dmd_x0+dmd_sig_w,
                  dmd_y0:dmd_y0+dmd_sig_h]
dmd_label_s = np.s_[dmd_x0:dmd_x0+dmd_sig_w,
                    dmd_y0+dmd_sig_h+dmd_R2_gap:dmd_y0+dmd_sig_h+dmd_R2_gap+dmd_label_h]
dmd_R1_s = np.s_[dmd_x0+dmd_sig_w+dmd_R1_gap:dmd_x0+dmd_sig_w+dmd_R1_gap+dmd_R1_w,
                 dmd_y0:dmd_y0+dmd_h]
dmd_R2_s = np.s_[dmd_x0:dmd_x0+dmd_sig_w,
                 dmd_y0+dmd_sig_h+dmd_R2_gap+dmd_label_h+dmd_R2_gap:dmd_y0+dmd_sig_h +
                 dmd_R2_gap+dmd_label_h+dmd_R2_gap+dmd_R2_h]
dmd_H1_s = np.s_[dmd_x0:dmd_x0+dmd_sig_w, dmd_y0+dmd_h-dmd_H1_h:dmd_y0+dmd_h]
dmd_V1_s = np.s_[dmd_x0+dmd_sig_w:dmd_x0+dmd_w, dmd_y0+dmd_h-dmd_H1_h:dmd_y0+dmd_h]

##############
# MEADOWLARK #
##############

slm2_resX = 1920
slm2_resY = 1152

slm2_w = 1920  # 1874  # int(slm_w * slm2_resX/slm_resX_actual)
slm2_h = 1152  # 860  # int(slm_h * slm2_resY/slm_resY_actual)

slm2_xc = slm2_resX//2
slm2_yc = slm2_resY//2

sig_y0, sig_y1 = 0, 900
R2_y0, R2_y1 = 910, 1000
lab_y0, lab_y1 = 1010, 1151

slm2_sig_w = 500
slm2_x0 = slm2_xc - slm2_sig_w//2

slm2_sig_h = sig_y1 - sig_y0
slm2_R2_h = R2_y1 - R2_y0
slm2_label_h = lab_y1 - lab_y0

slm2_sig_s = np.s_[slm2_x0:slm2_x0+slm2_sig_w, sig_y0:sig_y1]
slm2_R2_s = np.s_[slm2_x0:slm2_x0+slm2_sig_w, R2_y0:R2_y1]
slm2_label_s = np.s_[slm2_x0:slm2_x0+slm2_sig_w, lab_y0:lab_y1]

######################################

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

gr_ref = 2*np.pi*(-X+3*Y)/period+1e-5
gpu_gr_ref = cp.asarray(gr_ref, dtype='float32')

step = 0.00001
inv_sinc_LUT = np.load("./tools/inv_sinc_lut.npy")
gpu_inv_sinc_LUT = cp.asarray(inv_sinc_LUT, dtype='float32')

######################################

# create index arrays
x_SLM2 = slm2_w//2
y_SLM2 = slm2_h//2
X_array2 = np.arange(-x_SLM2, x_SLM2)
Y_array2 = np.arange(-y_SLM2, y_SLM2)
Y2, X2 = np.meshgrid(Y_array2, X_array2)

gr2 = 2*np.pi*(-X2+Y2)/period+1e-5
gpu_gr2 = cp.asarray(gr2, dtype='float32')

slm2_wf = np.load('./tools/slm2_wf.npy')
gpu_slm2_wf = cp.array(slm2_wf)

######################################

global n, m
global cols_to_del, rows_to_del, slm_block_w, slm_block_h
global rows_to_del2, slm2_block_h
global dmd_block_w, dmd_block_h, r, gpu_dmd_centers_x, gpu_dmd_centers_y
global ws, dmd_xc_shifted, y_blocks_multi_cp, bit_values_extended, o, u, e


def update_params(_n, _m, batch_size, num_frames):

    global n, m
    global cols_to_del, rows_to_del, slm_block_w, slm_block_h
    global rows_to_del2, slm2_block_h
    global dmd_block_w, dmd_block_h, r, gpu_dmd_centers_x, gpu_dmd_centers_y
    global ws, dmd_xc_shifted, y_blocks_multi_cp, bit_values_extended, o, u, e

    n = _n
    m = _m

    slm_block_w = int(slm_sig_w / n) + 1
    slm_block_h = int(slm_sig_h / m) + 1
    cols_to_del = np.linspace(0, slm_block_w*n - 1, (slm_block_w * n) - slm_sig_w).astype(np.int32)
    rows_to_del = np.linspace(0, slm_block_h*m - 1, (slm_block_h * m) - slm_sig_h).astype(np.int32)

    slm2_block_h = int(slm2_sig_h / m) + 1
    rows_to_del2 = np.linspace(0, slm2_block_h*m - 1, (slm2_block_h * m) - slm2_sig_h).astype(np.int32)

    slm_edges_x = np.linspace(0, slm_sig_w, n+1)
    slm_centers_x = np.array([(slm_edges_x[i]+slm_edges_x[i+1])/2 for i in range(n)]).astype(int)

    slm_edges_y = np.linspace(0, slm_sig_h, m+1)
    slm_centers_y = np.array([(slm_edges_y[i]+slm_edges_y[i+1])/2 for i in range(m)]).astype(int)

    dmd_block_w = int((slm_block_w-1) * dmd_w / slm_w)
    dmd_block_h = int((slm_block_h-1) * 0.9 * dmd_h / slm_h)

    if dmd_block_h % 2 == 0:
        dmd_block_h -= 1

    dmd_centers_x = (slm_centers_x * dmd_w / slm_w).astype(int)
    dmd_centers_y = (slm_centers_y * dmd_h / slm_h).astype(int)
    dmd_centers_x_grid, dmd_centers_y_grid = np.meshgrid(dmd_centers_x, dmd_centers_y)

    gpu_dmd_centers_x = cp.array(dmd_centers_x_grid)
    gpu_dmd_centers_y = cp.array(dmd_centers_y_grid)

    r = cp.array([int((-1) ** i * cp.ceil(i / 2)) for i in range(dmd_block_w+1)])
    ws, ns = cp.meshgrid(cp.arange(dmd_block_w+1), cp.arange(n), indexing='ij')
    dmd_xc_shifted = gpu_dmd_centers_x[0, :] - r[:, cp.newaxis]

    y_blocks = cp.zeros(dmd_sig_h, dtype='bool')
    for j in range(dmd_block_h // 2 + 1):
        y_blocks[gpu_dmd_centers_y[:, 0]-j] = 1
        y_blocks[gpu_dmd_centers_y[:, 0]+j] = 1

    y_blocks_multi = y_blocks[cp.newaxis, :].get().astype(np.bool_)
    y_blocks_multi_cp = cp.array(y_blocks_multi)

    bit_values = cp.zeros((3, 24))
    bit_values[0, :8] = cp.array([2**i for i in range(8)])
    bit_values[1, 8:16] = cp.array([2**i for i in range(8)])
    bit_values[2, 16:] = cp.array([2**i for i in range(8)])

    bit_values_extended = cp.repeat(bit_values[:, None, :, None, None], num_frames, axis=1).astype(cp.uint8)

    o = dmd_sig_w
    u = dmd_sig_h
    e = batch_size

    return dmd_block_w


def make_slm1_rgb(target,
                  R1_ampl=1., R1_phase=0,
                  R2_ampl=1., R2_phase=0,
                  label_ampl=0, label_phase=cp.pi):

    # global cols_to_del, rows_to_del, slm_block_w, slm_block_h

    target = np.flip(target, axis=1)

    gpu_A = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
    gpu_phi = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')

    aoi = np.repeat(target, slm_block_w, axis=0)
    aoi = np.repeat(aoi, slm_block_h, axis=1)
    aoi = np.delete(aoi, cols_to_del, 0)
    aoi = np.delete(aoi, rows_to_del, 1)

    A_aoi = np.abs(aoi.copy())
    phi_aoi = np.angle(aoi.copy())

    gpu_A[slm_sig_s] = cp.array(A_aoi)
    gpu_phi[slm_sig_s] = cp.array(phi_aoi)

    gpu_A[slm_label_s] = label_ampl
    gpu_phi[slm_label_s] = label_phase

    gpu_A[slm_R2_s] = R2_ampl
    gpu_phi[slm_R2_s] = R2_phase

    # gpu_A[slm_H1_s] = H1_ampl
    # gpu_phi[slm_H1_s] = H1_phase
    # gpu_A[slm_V1_s] = V1_ampl
    # gpu_phi[slm_V1_s] = V1_phase

    gpu_slm_wf_temp = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
    gpu_slm_wf_temp[slm_s] = gpu_slm_wf.copy()

    # wf correct
    gpu_phi[slm_sig_s] += gpu_slm_wf_temp[slm_sig_s]  # + gpu_wf_sig_vertical[:slm_sig_w, :slm_sig_h]
    gpu_phi[slm_label_s] += gpu_slm_wf_temp[slm_label_s]
    gpu_phi[slm_R2_s] += gpu_slm_wf_temp[slm_R2_s]
    # gpu_phi[slm_H1_s] += gpu_slm_wf_temp[slm_H1_s]
    # gpu_phi[slm_V1_s] += gpu_slm_wf_temp[slm_V1_s]

    gpu_A = cp.maximum(gpu_A, 1e-16, dtype='float32')
    gpu_A = cp.minimum(gpu_A, 1 - 1e-4, dtype='float32')
    gpu_A *= 0.99

    gpu_M = 1 - gpu_inv_sinc_LUT[(gpu_A/step).astype(cp.int)]
    gpu_F = gpu_phi + cp.pi*(1-gpu_M)
    gpu_sig = gpu_M * cp.mod(gpu_F+gpu_gr, 2*cp.pi)

    ##############

    gpu_A = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
    gpu_phi = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')

    gpu_A[slm_R1_s] = R1_ampl
    gpu_phi[slm_R1_s] = R1_phase

    # wf correct
    gpu_phi[slm_R1_s] += gpu_slm_wf[slm_sig_w:slm_w, :slm_h]

    gpu_A = cp.maximum(gpu_A, 1e-16, dtype='float32')
    gpu_A = cp.minimum(gpu_A, 1 - 1e-4, dtype='float32')
    gpu_A *= 0.99

    gpu_M = 1 - gpu_inv_sinc_LUT[(gpu_A/step).astype(cp.int)]
    gpu_F = gpu_phi + cp.pi*(1-gpu_M)
    gpu_ref = gpu_M*cp.mod(gpu_F+gpu_gr_ref, 2*cp.pi)

    ##############

    gpu_o = gpu_sig + gpu_ref

    # phase-GL LUT
    gpu_idx_phase = (gpu_o / d_phase).astype(int)
    gpu_idx_phase = cp.clip(gpu_idx_phase, 0, max_GL-1).astype(int)
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
    del gpu_idx_phase, gpu_o, gpu_M, gpu_A, gpu_F, gpu_phi

    cp._default_memory_pool.free_all_blocks()

    return gpu_color_array


def make_slm2_rgb(target, R2_ampl, R2_phase, label_ampl, label_phase):

    global slm2_block_h, rows_to_del2

    gpu_A = cp.zeros((slm2_resX, slm2_resY), dtype='float32')
    gpu_phi = cp.zeros((slm2_resX, slm2_resY), dtype='float32')

    aoi = np.repeat(target, slm2_sig_w, axis=1)
    aoi = np.repeat(aoi, slm2_block_h, axis=0)
    aoi = np.delete(aoi, rows_to_del2, axis=0).T

    print(aoi.shape)

    A_aoi = np.abs(aoi.copy())
    phi_aoi = np.angle(aoi.copy())

    gpu_A[slm2_sig_s] = cp.array(A_aoi)
    gpu_phi[slm2_sig_s] = cp.array(phi_aoi) + gpu_slm2_wf.copy()[slm2_sig_s]

    gpu_A[slm2_R2_s] = R2_ampl
    gpu_phi[slm2_R2_s] = R2_phase

    gpu_A[slm2_label_s] = label_ampl
    gpu_phi[slm2_label_s] = label_phase

    gpu_A = cp.maximum(gpu_A, 1e-16, dtype='float32')
    gpu_A = cp.minimum(gpu_A, 1 - 1e-4, dtype='float32')
    gpu_A *= 0.99

    gpu_M = 1 - gpu_inv_sinc_LUT[(gpu_A/step).astype(cp.int)]
    gpu_F = gpu_phi + cp.pi*(1-gpu_M)
    gpu_o = gpu_M*cp.mod(gpu_F+gpu_gr2, 2*cp.pi)

    # eventually replace with proper GL-phase LUT
    gpu_GL = (gpu_o*255/(2*cp.pi)).astype(cp.uint8)

    gpu_out = gpu_GL[...,None].repeat(3, axis=-1)

    return gpu_out


def make_dmd_image(arr, R1_ampl, R2_ampl, label_ampl):

    global n, m, dmd_block_w, dmd_block_h, r, gpu_dmd_centers_x, gpu_dmd_centers_y

    arr = cp.array(arr*dmd_block_w)

    ws_local, ms_local, ns_local = cp.meshgrid(cp.arange(dmd_block_w + 1), cp.arange(m), cp.arange(n),
                                               indexing='ij')
    dmd_xcc = gpu_dmd_centers_x - r[:, cp.newaxis, cp.newaxis]
    dmd_ycc = gpu_dmd_centers_y[cp.newaxis, ...].repeat(dmd_block_w + 1, axis=0)

    target = arr[..., cp.newaxis].repeat(dmd_block_w + 1, axis=-1)

    mask = (target > cp.arange(dmd_block_w + 1))
    mask = mask.transpose(2, 1, 0)

    mapped = cp.zeros((dmd_block_w + 1, dmd_sig_h, dmd_sig_w), dtype='bool')

    for j in range(dmd_block_h // 2):
        mapped[ws_local, dmd_xcc - j, dmd_xc] = mask
        mapped[ws_local, dmd_ycc + j + 1, dmd_xc] = mask

    out = mapped.sum(axis=0).astype(cp.uint8)
    out = cp.flip(out, 0).T

    img = cp.zeros((dmd_w, dmd_h))
    img[dmd_sig_s] = out.copy()

    if R1_ampl > 0:
        img[dmd_R1_s] = 1
    if R2_ampl > 0:
        img[dmd_R2_s] = 1
    if label_ampl > 0:
        img[dmd_label_s] = 1

    rgb = (img.T)[..., None].repeat(4, axis=2)
    rgb *= 255
    rgb[..., -1] = 255

    return rgb.astype(cp.uint8)


def make_dmd_batch(vecs, R1_ampl, R2_ampl, label_ampl, batch_size, num_frames):

    global dmd_block_w, ws, dmd_xc_shifted, y_blocks_multi_cp, bit_values_extended, o, u, e

    def find_1d_pattern(vec):
        target = vec[cp.newaxis, :].astype(cp.uint8).repeat(dmd_block_w + 1, axis=0)
        mask = (target > cp.arange(dmd_block_w + 1)[:, cp.newaxis])
        mapped = cp.zeros((dmd_block_w + 1, dmd_sig_w), dtype='bool')
        mapped[ws, dmd_xc_shifted] = mask
        out = mapped.sum(axis=0).astype(cp.bool)
        return cp.flip(out)

    outxs = cp.empty((batch_size, dmd_sig_w), dtype='bool')
    for indx, vec in enumerate(vecs):
        outxs[indx, :] = find_1d_pattern(vec*dmd_block_w)

    imgs = cp.einsum('eo,eu->eou', outxs, y_blocks_multi_cp).astype(cp.bool)

    imgs = imgs.reshape((num_frames, 24, dmd_sig_w, dmd_sig_h)).astype(cp.uint8)[None, ...].repeat(3, axis=0)
    imgs *= bit_values_extended
    imgs = imgs.sum(axis=2)
    imgs = cp.transpose(imgs, (1, 3, 2, 0))
    imgs = cp.flip(imgs, axis=1)
    del outxs

    out = cp.zeros((num_frames, dmd_h, dmd_w, 3), dtype=cp.uint8)

    out[np.s_[:, dmd_sig_s[1], dmd_sig_s[0], :]] = imgs.copy()

    if R1_ampl > 0:
        out[np.s_[:, dmd_R1_s[1], dmd_R1_s[0], :]] = 255
    if R2_ampl > 0:
        out[np.s_[:, dmd_R2_s[1], dmd_R2_s[0], :]] = 255
    if label_ampl > 0:
        out[np.s_[:, dmd_label_s[1], dmd_label_s[0], :]] = 255

    return out
