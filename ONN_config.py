import time
import os
import numpy as np
import cupy as cp
from scipy.optimize import curve_fit
from termcolor import colored
from pypylon import pylon
from glumpy import app
from glumpy.app import clock
from glumpy_display import setup, window_on_draw
from slm_display import SLMdisplay
from make_slm1_image import make_slm_rgb, make_dmd_image, make_dmd_batch, update_params

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class CaptureProcess(pylon.ImageEventHandler):

    def __init__(self):
        super().__init__()

        self.frames = []

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            image = grab_result.GetArray()

            if image.max() > 8:
                self.frames.append(image)

            self.frames = self.frames[-1001:]


class Controller:

    def __init__(self,  n, m, ref_spot, ref_block_val, batch_size, num_frames, is_complex, mout,
                 ampl_norm_val, scale_guess, ref_guess, meas_type, layers, ref_on, label_block_on):

        self.n = n
        self.m = m
        self.ref_spot = ref_spot
        self.ref_block_val = ref_block_val
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.is_complex = is_complex
        self.mout = mout
        self.ampl_norm_val = ampl_norm_val
        self.scale_guess = scale_guess
        self.ref_guess = ref_guess
        self.meas_type = meas_type
        self.layers = layers
        self.ref_on = ref_on
        self.label_block_on = label_block_on

        self.y_centers = np.load('./tools/y_centers_list.npy')
        self.x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
        self.complex_output_ratios = np.load('./tools/complex_output_ratios.npy')

        self.dmd_block_w = update_params(self.n, self.m, self.ref_spot, self.ref_block_val, self.batch_size,
                                         self.num_frames, self.is_complex, self.label_block_on)

        if self.meas_type == 'reals':

            actual_uppers_arr_256 = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/"
                                            "ONN/tools/actual_uppers_arr_256.npy")
            # actual_uppers_arr_256[:, :, ref_spot] = actual_uppers_arr_256[:, :, ref_spot+1]
            self.uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
            # uppers1_ann = np.delete(uppers1_nm, ref_spot, 1)
            self.gpu_actual_uppers_arr = cp.asarray(actual_uppers_arr_256)

        else:

            actual_uppers_arr_128_flat = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects"
                                                 "/ONN/tools/actual_uppers_arr_128_flat.npy")
            actual_uppers_arr_128_flat /= np.max(actual_uppers_arr_128_flat)
            self.uppers1_nm = actual_uppers_arr_128_flat[-1, ...].copy()
            self.gpu_actual_uppers_arr = cp.asarray(actual_uppers_arr_128_flat)

        if m != mout:
            self.uppers1_ann = self.uppers1_nm.reshape(n, mout, m//mout).mean(axis=-1)

        self.slm = SLMdisplay()

        self.backend = app.use('glfw')

        self.window = app.Window(1920, 1080, fullscreen=0, decoration=0)
        self.window.set_position(-1920-1920-2560, 0)
        self.window.activate()
        self.window.show()

        self.dmd_clock = clock.Clock()

        @self.window.event
        def on_draw(dt):
            window_on_draw(self.window, self.screen, self.cuda_buffer, self.cp_arr)
            self.frame_count += 1
            self.cp_arr = self.target_frames[self.frame_count % len(self.target_frames)]

        self.screen, self.cuda_buffer, self.context = setup(1920, 1080)

        self.null_frame = self.dmd_one_frame(np.zeros((self.n, self.m)), ref=0)[0]
        self.null_frames = [self.null_frame for _ in range(10)]

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        print("Using device ", self.camera.GetDeviceInfo().GetModelName())

        pylon.FeaturePersistence.Load("./tools/pylon_settings.pfs", self.camera.GetNodeMap())
        # register the background handler and start grabbing using background pylon thread
        self.capture = CaptureProcess()
        self.camera.RegisterImageEventHandler(self.capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
        time.sleep(1)

        self.target_frames = None
        self.fc = None
        self.cp_arr = None
        self.frame_count = None
        self.frames = None
        self.ampls = None
        self.z1s = None
        self.norm_params = None

        self.init_dmd()

    def dmd_one_frame(self, arr, ref):
        img = make_dmd_image(arr, ref=ref, ref_block_val=self.ref_block_val, dmd_block_w=self.dmd_block_w)
        return [img]

    def init_dmd(self):

        self.target_frames = cp.zeros((self.num_frames + 4, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[..., -1] = 255
        self.fc = self.target_frames.shape[0] - 1
        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        for _ in range(5):
            self.capture.frames = []
            app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
            time.sleep(0.1)

    def run_batch(self, vecs_in, labels=None, normalise=False):

        t0 = time.perf_counter()

        self.target_frames[2:-2, :, :, :-1] = make_dmd_batch(vecs_in, ref=self.ref_on, ref_block_val=self.ref_block_val,
                                                             batch_size=self.batch_size, num_frames=self.num_frames,
                                                             label_block_on=self.label_block_on, labels=labels)

        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        self.capture.frames = []
        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.1)

        self.frames = np.array(self.capture.frames.copy())
        self.ampls = self.find_spot_ampls(self.frames.copy())

        print('num frames = ', self.frames.shape)

        np.save('./tools/frames_temp.npy', self.frames)

        success = self.check_num_frames(self.batch_size)
        if success:
            self.process_ampls(normalise)
        else:
            self.z1s = np.full((self.batch_size, self.mout), np.nan)

        t1 = time.perf_counter()

        print('batch time: {:.3f}'.format(t1 - t0))
        return success

    def find_spot_ampls(self, arrs):
        mask = arrs < 3
        arrs -= 2
        arrs[mask] = 0

        def spot_s(i):
            return np.s_[:, self.y_centers[i] - 2:self.y_centers[i] + 3,
                         self.x_edge_indxs[2 * i]:self.x_edge_indxs[2 * i + 1]]

        spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.m + 1)])
        # spot_powers = cp.random.randint(0, 256, (240, self.m+1)).T

        spot_ampls = cp.sqrt(spot_powers)

        spot_ampls = cp.flip(spot_ampls, axis=0)

        ratio = spot_ampls[self.ref_spot, :] / spot_ampls[self.ref_spot + 1, :]

        spot_ampls[self.ref_spot + 1:, :] *= ratio[None, :]

        spot_ampls = np.delete(spot_ampls.get(), self.ref_spot + 1, 0)

        return spot_ampls.T

    def update_slm(self, arr, lut=False, noise_arr_A=None, noise_arr_phi=None):

        if arr.shape[1] == self.m - 1:
            arr = np.insert(arr, self.ref_spot, np.zeros(self.n), 1)

        if self.is_complex:
            arr = np.repeat(arr.copy(), 4, axis=1) * self.complex_output_ratios.copy()[None, :]

            if lut:
                gpu_arr = cp.abs(cp.asarray(arr.copy()))
                map_indx = cp.argmin(cp.abs(self.gpu_actual_uppers_arr - gpu_arr), axis=0)
                arr_A = cp.linspace(0., 1., 128)[map_indx]
            else:
                arr_A = cp.abs(cp.asarray(arr.copy()))

            arr_phi = cp.angle(cp.array(arr.copy()))

            if noise_arr_A is not None:
                arr_A += cp.array(noise_arr_A)

            if noise_arr_phi is not None:
                arr_phi += cp.array(noise_arr_phi)

            arr_out = arr_A * cp.exp(1j * arr_phi)

        else:

            arr = np.repeat(arr.copy(), 4, axis=1) * self.complex_output_ratios.copy()[None, :]

            if noise_arr_A is not None:
                arr += noise_arr_A.copy()

            if lut:
                gpu_arr = cp.asarray(arr.copy())
                map_indx = cp.argmin(cp.abs(self.gpu_actual_uppers_arr - gpu_arr), axis=0)
                arr_out = cp.linspace(-1., 1., 256)[map_indx]
            else:
                arr_out = cp.asarray(arr.copy())

            if self.ref_on:
                arr_out[:, self.ref_spot] = self.ampl_norm_val

        arr_out = np.flip(arr_out.get(), axis=1)
        img = make_slm_rgb(arr_out, self.ref_block_val)
        self.slm.updateArray(img)

    def check_num_frames(self, true_size):

        if self.ampls.shape[0] == true_size:

            meas = self.ampls.copy().reshape((self.num_frames, true_size // self.num_frames, self.m))
            diffs = np.abs(np.array([meas[kk + 1, :, self.m // 3] - meas[kk, :, self.m // 3]
                                     for kk in range(self.num_frames - 1)])).mean(axis=1)
            diffs /= diffs.max()
            repeats = (diffs < 0.25).sum() > 0

            if repeats:
                print(colored('repeated frames', 'red'))
                return False

            else:
                return True

        else:
            print(colored('wrong num frames: {}'.format(self.ampls.shape[0]), 'red'))
            return False

    def process_ampls(self, normalise=False):

        if self.meas_type == 'reals':

            ampls = self.ampls.copy()
            if self.m != self.mout:
                ampls = ampls.reshape((self.ampls.shape[0], self.mout, self.m // self.mout)).mean(axis=-1)

            z1s = (ampls - self.ref_guess) * self.scale_guess

        elif self.meas_type == 'complex':

            Iall = self.ampls.copy() ** 2
            I0 = Iall[:, 0::4].copy()
            I1 = Iall[:, 1::4].copy()
            I2 = Iall[:, 2::4].copy()
            I3 = Iall[:, 3::4].copy()
            Xmeas = (I0 - I2) / self.scale_guess
            Ymeas = (I1 - I3) / self.scale_guess
            z1s = Xmeas + (1j * Ymeas)

        elif self.meas_type == 'complex_power':
            Imeas = self.ampls.copy() ** 2
            Imeas = Imeas.reshape(Imeas.shape[0], 10, 4).mean(axis=-1)
            Imeas *= self.scale_guess
            z1s = Imeas.copy()

        else:
            raise ValueError

        if normalise:
            z1s_norm = self.normalise(z1s)
            self.z1s = z1s_norm
        else:
            self.z1s = z1s

    def find_norm_params(self, theory, measured):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory.shape[1] == measured.shape[1]

        if self.meas_type == 'complex':

            real_norm_params = np.array([curve_fit(line, np.real(theory[:, j]), np.real(measured[:, j]))[0]
                                         for j in range(theory.shape[1])])
            imag_norm_params = np.array([curve_fit(line, np.imag(theory[:, j]), np.imag(measured[:, j]))[0]
                                         for j in range(theory.shape[1])])

            self.norm_params = real_norm_params + (1j * imag_norm_params)

        else:
            self.norm_params = np.array([curve_fit(line, theory[:, j], measured[:, j])[0]
                                    for j in range(theory.shape[1])])

    def update_norm_params(self, theory, measured):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory.shape[1] == measured.shape[1]

        if self.meas_type == 'complex':

            real_norm_params_adjust = np.array([curve_fit(line, np.real(theory[:, j]), np.real(measured[:, j]))[0]
                                                for j in range(self.mout)])
            real_norm_params = np.real(self.norm_params.copy())
            real_norm_params[:, 1] += real_norm_params[:, 0].copy() * real_norm_params_adjust[:, 1].copy()
            real_norm_params[:, 0] *= real_norm_params_adjust[:, 0].copy()

            imag_norm_params_adjust = np.array([curve_fit(line, np.imag(theory[:, j]), np.imag(measured[:, j]))[0]
                                                for j in range(self.mout)])
            imag_norm_params = np.imag(self.norm_params.copy())
            imag_norm_params[:, 1] += imag_norm_params[:, 0].copy() * imag_norm_params_adjust[:, 1].copy()
            imag_norm_params[:, 0] *= imag_norm_params_adjust[:, 0].copy()

            self.norm_params = real_norm_params + (1j * imag_norm_params)

        else:

            norm_params_adjust = np.array([curve_fit(line, theory[:, j], measured[:, j])[0]
                                           for j in range(self.mout)])
            self.norm_params[:, 1] += self.norm_params[:, 0].copy() * norm_params_adjust[:, 1].copy()
            self.norm_params[:, 0] *= norm_params_adjust[:, 0].copy()

    def normalise(self, z1s_in):

        if self.meas_type == 'complex':

            Zreals = (np.real(z1s_in).copy() - np.real(self.norm_params.copy())[:, 1]) \
                     / np.real(self.norm_params.copy())[:, 0]
            Zimags = (np.imag(z1s_in).copy() - np.imag(self.norm_params.copy())[:, 1]) \
                     / np.imag(self.norm_params.copy())[:, 0]
            z1s = Zreals + (1j * Zimags)

        else:
            z1s = (z1s_in - self.norm_params[:, 1].copy()) / self.norm_params[:, 0].copy()

        return z1s

    def close(self):
        self.camera.Close()
        # imageWindow.Close()
        self.context.pop()
        app.quit()
        print('\nfinished\n')
        os._exit(0)
