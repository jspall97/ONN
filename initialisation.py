import time
import sys
import signal
import numpy as np
import cupy as cp
from glumpy import app
from make_dmd_image import make_dmd_rgb, make_dmd_image, make_dmd_rgb_multi
from MNIST import make_dmd_rgb as make_dmd_rgb_new
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm1_image import make_slm_rgb
from multiprocessing import Process, Pipe
from pylon import run_camera, view_camera, Camera
import threading

# aoi_w, aoi_h = 1024, 120

# area_height = 4
# half_height_200 = area_height // 2  # mean box height
#
# y_center_indxs_200 = np.load('./tools/y_center_indxs_200.npy')
# x_edge_indxs_200 = np.load('./tools/x_edge_indxs_200.npy')

#
# try:
#     for j in range(m):
#         rects_200[j].remove()
# except (ValueError, NameError) as e:
#     pass
#
# rects_200 = {}
# for j in range(m):
#     rect_w = x_edge_indxs_200[(2 * j) + 1] - x_edge_indxs_200[2 * j]
#     rect_h = area_height
#     xl = x_edge_indxs_200[2 * j]
#     yt = y_center_indxs_200[j] - half_height_200
#
#     rect = patches.Rectangle((xl, yt), rect_w, rect_h, linewidth=1, edgecolor='r', facecolor='none')
#     rects_200[j] = cam_axes_200.add_patch(rect)
#
# try:
#     for j in range(m):
#         rects_200[j].remove()
# except (ValueError, NameError) as e:
#     pass


#def find_spot_ampls_200(arr):
#
#     def spot_s_200(i):
#         y_center_i = y_center_indxs_200[i]
#         if half_height_200 == 0:
#             return np.s_[x_edge_indxs_200[2 * i]:x_edge_indxs_200[2 * i + 1], y_center_i]
#         else:
#             return np.s_[x_edge_indxs_200[2 * i]:x_edge_indxs_200[2 * i + 1],
#                          y_center_i - half_height_200:y_center_i + half_height_200 + 1]
#
#     spots_dict = {}
#
#     for spot_num in range(m):
#         spot = arr[spot_s_200(spot_num)]
#         spots_dict[spot_num] = spot
#
#     spot_powers = np.array([spots_dict[i].mean() for i in range(m)])
#
#     spot_ampls = np.sqrt(spot_powers)
#
#     return np.array(spot_ampls)



n = 121
m = 52


# def dmd_one_frame(arr):
#     img = make_dmd_image(arr)
#     frame = make_dmd_rgb([img for _ in range(24)])
#     return frame

zero_frame = make_dmd_image(np.zeros((n, m)))

dmd_cols = []
for i in range(n):
    print(i)
    dmd_col = np.zeros((n, m))
    dmd_col[i, :] = 1
    dmd_col_img = make_dmd_image(dmd_col)

    for _ in range(24):
        dmd_cols.append(dmd_col_img)

    dmd_cols_i = make_dmd_rgb_multi(dmd_cols)
    cp.save('./tools/dmd_imgs/cols/col_array_{}'.format(i), cp.array(dmd_cols_i))

    del dmd_cols_i, dmd_col_img

# for i in range(24 - (n%24)):
#     dmd_cols.append(zero_frame)

print(len(dmd_cols))










