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

            if image.max() > 0:  # 8
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

        self.y_centers = np.load('./tools/y_centers_list.npy')
        self.x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
        self.complex_output_ratios = np.load('./tools/complex_output_ratios.npy')

        self.dmd_block_w = update_params(self.n, self.m, self.batch_size, self.num_frames)

        actual_uppers_arr_256 = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/"
                                        "ONN/tools/actual_uppers_arr_256.npy")
        self.uppers1_nm = actual_uppers_arr_256[-1, ...].copy()
        self.gpu_actual_uppers_arr = cp.asarray(actual_uppers_arr_256)

        # self.slm = SLMdisplay(-1920-2560, 0, 1920/2, 1080/2, 'SLM 1', True)
        self.slm = SLMdisplay(-1920, 1920/2, 1080, 'SLM 1',
                              -1920/2, 1920/2, 1080, 'SLM 2',
                              True)

        self.backend = app.use('glfw')
        self.window = app.Window(1920, 1080, fullscreen=0, decoration=0)
        self.window.set_position(-1920-1920-2560, 0)
        # self.window.set_position(0, 0)
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

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        print("Using device ", self.camera.GetDeviceInfo().GetModelName())

        pylon.FeaturePersistence.Load("./tools/pylon_settings_burst.pfs", self.camera.GetNodeMap())
        # register the background handler and start grabbing using background pylon thread
        self.capture = CaptureProcess()
        self.camera.RegisterImageEventHandler(self.capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
        self.camera.AcquisitionMode.SetValue("Continuous")
        self.camera.TriggerSelector.SetValue("FrameBurstStart")
        self.camera.TriggerMode.SetValue("On")
        self.camera.TriggerSource.SetValue("Line1")
        self.camera.TriggerActivation.SetValue("RisingEdge")
        self.camera.AcquisitionFrameRate.SetValue(1440)
        self.camera.AcquisitionBurstFrameCount.SetValue(24)

        self.capture.frames = []
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

        print('setup complete')

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

    def run_batch(self, vecs_in, normalise=False):

        t0 = time.perf_counter()

        self.target_frames[2:-2, :, :, :-1] = make_dmd_batch(vecs_in, self.R1_ampl, self.R2_ampl, self.label_ampl,
                                                             self.batch_size, self.num_frames)

        self.cp_arr = self.target_frames[0]
        self.frame_count = 0

        self.capture.frames = []
        app.run(clock=self.dmd_clock, framerate=0, framecount=self.fc)
        time.sleep(0.1)

        self.frames = np.array(self.capture.frames.copy())
        self.ampls = self.find_spot_ampls(self.frames.copy())

        print('num frames = ', self.frames.shape)
        np.save('./tools/frames_temp.npy', self.frames)

        success = self.check_num_frames(true_size=self.batch_size)
        if success:
            self.process_ampls(normalise=normalise)
        else:
            self.z1s = np.full((self.batch_size, self.m), np.nan)

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

        # spot_powers = cp.array([arrs[spot_s(i)].mean(axis=(1, 2)) for i in range(self.m + 1)])
        spot_powers = cp.random.randint(0, 256, (240, self.m)).T

        spot_ampls = cp.sqrt(spot_powers)
        spot_ampls = cp.flip(spot_ampls, axis=0).T

        return spot_ampls.get()

    def create_slm1(self, arr, lut):

        if lut:
            gpu_arr = cp.asarray(arr.copy())
            map_indx = cp.argmin(cp.abs(self.gpu_actual_uppers_arr - gpu_arr), axis=0)
            arr_out = cp.linspace(-1., 1., 256)[map_indx]
        else:
            arr_out = cp.asarray(arr.copy())

        arr_out = np.flip(arr_out.get(), axis=1)

        img = make_slm1_rgb(arr_out, R1_ampl=self.R1_ampl, R1_phase=0, R2_ampl=1., R2_phase=0.,
                            label_ampl=1., label_phase=0.)

        return img

    def create_slm2(self, arr, lut):
        img = make_slm2_rgb(arr, R2_ampl=self.R2_ampl, R2_phase=0,
                            label_ampl=self.label_ampl, label_phase=cp.pi)

        return img

    def update_slms(self, slm1_arr, slm2_arr, slm1_lut=False, slm2_lut=False):

        img1 = self.create_slm1(slm1_arr, slm1_lut)
        img2 = self.create_slm2(slm2_arr, slm2_lut)
        self.slm.updateArray(img1, img2)

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

        ampls = self.ampls.copy()

        z1s = (ampls - self.ref_guess) * self.scale_guess

        if normalise:
            z1s_norm = self.normalise(z1s)
            self.z1s = z1s_norm
        else:
            self.z1s = z1s

    def find_norm_params(self, theory, measured):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory.shape[1] == measured.shape[1]

        self.norm_params = np.array([curve_fit(line, theory[:, j], measured[:, j])[0]
                                    for j in range(theory.shape[1])])

    def update_norm_params(self, theory, measured):

        def line(x, grad, c):
            return (grad * x) + c

        assert theory.shape[1] == measured.shape[1]

        norm_params_adjust = np.array([curve_fit(line, theory[:, j], measured[:, j])[0]
                                       for j in range(self.m)])
        self.norm_params[:, 1] += self.norm_params[:, 0].copy() * norm_params_adjust[:, 1].copy()
        self.norm_params[:, 0] *= norm_params_adjust[:, 0].copy()

    def normalise(self, z1s_in):
        return (z1s_in - self.norm_params[:, 1].copy()) / self.norm_params[:, 0].copy()

    def close(self):
        self.camera.Close()
        # imageWindow.Close()
        self.context.pop()
        app.quit()
        print('\nfinished\n')
        os._exit(0)
