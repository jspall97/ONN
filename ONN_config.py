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
from make_slm_image_2d import make_slm1_rgb, make_slm2_rgb,\
                              make_dmd_image, make_dmd_batch, update_params

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class CaptureProcess(pylon.ImageEventHandler):

    def __init__(self):
        super().__init__()

        self.frames = []

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            image = grab_result.GetArray()

            if image.max() > 8:  # 8
                self.frames.append(image)

            self.frames = self.frames[-1001:]


class Controller:

    def __init__(self,  n, m, batch_size, num_frames, scale_guess, ref_guess,
                 R1_ampl, R2_ampl, label_ampl):

        self.n = n
        self.m = m

        self.batch_size = batch_size
        self.num_frames = num_frames

        self.scale_guess = scale_guess
        self.ref_guess = ref_guess

        self.R1_ampl = R1_ampl
        self.R2_ampl = R2_ampl
        self.label_ampl = label_ampl

        #######
        # SLM #
        #######

        self.y_centers = np.load('./tools/y_centers_list.npy')
        self.x_edge_indxs = np.load('./tools/x_edge_indxs.npy')

        self.dmd_block_w = update_params(self.n, self.m, self.batch_size, self.num_frames)

        actual_uppers_arr_256 = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/"
                                        "ONN/tools/actual_uppers_arr_256.npy")
        self.uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
        self.gpu_actual_uppers_arr = cp.asarray(actual_uppers_arr_256)

        self.slm = SLMdisplay(-1920-2560, 1920, 1080, 'SLM 1',
                              2560, 1920, 1152, 'SLM 2',
                              True)

        #######
        # DMD #
        #######

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

        self.null_frame = make_dmd_image(np.zeros((self.n, self.m)), 0., 0., 0.)
        self.null_frames = [self.null_frame for _ in range(10)]

        ###########
        # CAMERAS #
        ###########

        tlfactory = pylon.TlFactory.GetInstance()
        devices = tlfactory.EnumerateDevices()
        assert len(devices) == 2
        self.cameras = pylon.InstantCameraArray(2)
        for i, camera in enumerate(self.cameras):
            camera.Attach(tlfactory.CreateDevice(devices[i]))

        pylon.FeaturePersistence.Load("./tools/pylon_settings_burst.pfs", self.cameras[0].GetNodeMap())
        pylon.FeaturePersistence.Load("./tools/pylon_settings_burst.pfs", self.cameras[1].GetNodeMap())

        for camera in self.cameras:
            camera.AcquisitionMode.SetValue("Continuous")
            camera.TriggerSelector.SetValue("FrameBurstStart")
            camera.TriggerMode.SetValue("On")
            camera.TriggerSource.SetValue("Line1")
            camera.TriggerActivation.SetValue("RisingEdge")
            camera.AcquisitionFrameRate.SetValue(1440)
            camera.AcquisitionBurstFrameCount.SetValue(24)

        self.capture1 = CaptureProcess()
        self.capture2 = CaptureProcess()

        self.cameras[0].RegisterImageEventHandler(self.capture1, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
        self.cameras[1].RegisterImageEventHandler(self.capture2, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)

        self.capture1.frames = []
        self.capture2.frames = []
        self.cameras.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
        time.sleep(1)

        self.target_frames = None
        self.fc = None
        self.cp_arr = None
        self.frame_count = None

        self.frames1 = None
        self.ampls1 = None
        self.z1s = None
        self.norm_params1 = None

        self.frames2 = None
        self.ampls2 = None
        self.z2s = None
        self.norm_params2 = None

        self.init_dmd()

        print('setup complete')

    def init_dmd(self):

        self.target_frames = cp.zeros((self.num_frames + 4, 1080, 1920, 4), dtype=cp.uint8)
        self.target_frames[..., -1] = 255
        self.fc = self.target_frames.shape[0] - 1
        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        for _ in range(5):
            self.capture1.frames = []
            self.capture2.frames = []
            app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
            time.sleep(0.1)

    def run_batch(self, vecs_in, labels=None, normalisation=False):

        t0 = time.perf_counter()

        self.target_frames[2:-2, :, :, :-1] = make_dmd_batch(vecs_in, labels, self.R1_ampl, self.R2_ampl, self.label_ampl,
                                                             self.batch_size, self.num_frames)

        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        self.capture1.frames = []
        self.capture2.frames = []
        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.1)

        self.frames1 = np.array(self.capture1.frames)
        self.frames2 = np.array(self.capture2.frames)
        self.ampls1 = self.find_spot_ampls1(self.frames1.copy())
        self.ampls2 = self.find_spot_ampls2(self.frames2.copy())

        print('num frames 1 = ', self.frames1.shape)
        print('num frames 2 = ', self.frames2.shape)
        np.save('./tools/frames1_temp.npy', self.frames1)
        np.save('./tools/frames2_temp.npy', self.frames2)

        success = self.check_num_frames(true_size=self.batch_size)
        self.process_ampls(success, normalisation)

        t1 = time.perf_counter()

        print('batch time: {:.3f}'.format(t1 - t0))
        return success

    def find_spot_ampls1(self, arrs):
        mask = arrs < 3
        arrs -= 2
        arrs[mask] = 0

        def spot_s(i):
            return np.s_[:, self.y_centers[i] - 2:self.y_centers[i] + 3,
                         self.x_edge_indxs[2 * i]:self.x_edge_indxs[2 * i + 1]]

        # spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.m + 1)])
        spot_powers = cp.random.randint(0, 256, (240, self.m)).T

        spot_ampls = cp.sqrt(spot_powers)
        spot_ampls = cp.flip(spot_ampls, axis=0).T

        return spot_ampls.get()

    def find_spot_ampls2(self, arrs):
        spot_powers = cp.random.randint(0, 256, (240, 1))
        return spot_powers.get()

    def check_num_frames(self, true_size):

        if (self.ampls1.shape[0] == true_size) and (self.ampls2.shape[0] == true_size):

            def check_repeats(ampls):
                meas = ampls.reshape((self.num_frames, true_size // self.num_frames, self.m))
                diffs = np.abs(np.array([meas[kk + 1, :, self.m // 3] - meas[kk, :, self.m // 3]
                                         for kk in range(self.num_frames - 1)])).mean(axis=1)
                diffs /= diffs.max()
                repeats = (diffs < 0.25).sum() > 0
                return repeats

            if check_repeats(self.ampls1.copy()) or check_repeats(self.ampls2.copy()):
                print(colored('repeated frames', 'red'))
                return False
            else:
                return True

        else:
            print(colored(f'wrong num frames: {self.ampls1.shape[0]} and {self.ampls2.shape[0]}', 'red'))
            return False

    def update_slm1(self, arr, lut):

        if lut:
            gpu_arr = cp.asarray(arr.copy())
            map_indx = cp.argmin(cp.abs(self.gpu_actual_uppers_arr - gpu_arr), axis=0)
            arr_out = cp.linspace(-1., 1., 256)[map_indx]
        else:
            arr_out = cp.asarray(arr.copy())

        arr_out = np.flip(arr_out.get(), axis=1)

        img = make_slm1_rgb(arr_out, R1_ampl=self.R1_ampl, R1_phase=0)

        self.slm.updateArray_slm1(img)

    def update_slm2(self, arr, lut):

        img = make_slm2_rgb(arr, R2_ampl=self.R2_ampl, R2_phase=0,
                            label_ampl=self.label_ampl, label_phase=cp.pi)

        self.slm.updateArray_slm2(img)

    def find_norm_params(self, theory1, measured1, theory2, measured2):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory1.shape[1] == measured1.shape[1]
        assert theory2.shape[1] == measured2.shape[1]

        self.norm_params1 = np.array([curve_fit(line, theory1[:, j], measured1[:, j])[0]
                                     for j in range(theory1.shape[1])])

        self.norm_params2 = curve_fit(line, theory2[:, 0], measured2[:, 0])[0]

    def update_norm_params(self, theory1, measured1, theory2, measured2):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory1.shape[1] == measured1.shape[1]
        assert theory2.shape[1] == measured2.shape[1]

        norm_params_adjust1 = np.array([curve_fit(line, theory1[:, j], measured1[:, j])[0]
                                       for j in range(self.m)])
        self.norm_params1[:, 1] += self.norm_params1[:, 0].copy() * norm_params_adjust1[:, 1].copy()
        self.norm_params1[:, 0] *= norm_params_adjust1[:, 0].copy()

        norm_params_adjust2 = curve_fit(line, theory2[:, 0], measured2[:, 0])[0]
        self.norm_params2[1] += self.norm_params2[0].copy() * norm_params_adjust2[1].copy()
        self.norm_params2[0] *= norm_params_adjust2[0].copy()

    def process_ampls(self, success, normalise):
        if success:
            if normalise:
                self.z1s = (self.ampls1.copy() - self.norm_params1[:, 1].copy()) / self.norm_params1[:, 0].copy()
                self.z2s = (self.ampls2.copy() - self.norm_params2[1].copy()) / self.norm_params2[0].copy()
            else:
                self.z1s = self.ampls1.copy()
                self.z2s = self.ampls2.copy()
        else:
            self.z1s = np.full((self.batch_size, self.m), np.nan)
            self.z2s = np.full((self.batch_size, 1), np.nan)

    def close(self):
        self.cameras.Close()
        # imageWindow.Close()
        self.context.pop()
        app.quit()
        print('\nfinished\n')
        os._exit(0)
