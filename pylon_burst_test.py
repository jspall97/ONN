import time
import numpy as np
import cupy as cp
from pypylon import pylon


class CaptureProcess(pylon.ImageEventHandler):

    def __init__(self):
        super().__init__()

        self.frames = []

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():

            image = grab_result.GetArray()

            # if image.max() > 8:
            self.frames.append(image)

            self.frames = self.frames[-1001:]


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

print("Using device ", camera.GetDeviceInfo().GetModelName())

pylon.FeaturePersistence.Load("./tools/pylon_settings_burst.pfs", camera.GetNodeMap())
# register the background handler and start grabbing using background pylon thread
capture = CaptureProcess()
camera.RegisterImageEventHandler(capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)

time.sleep(1)

camera.AcquisitionMode.SetValue("Continuous")
camera.TriggerSelector.SetValue("FrameBurstStart")
camera.TriggerMode.SetValue("On")
camera.TriggerSource.SetValue("Line1")
camera.TriggerActivation.SetValue("RisingEdge")
camera.AcquisitionFrameRate.SetValue(1440)
camera.AcquisitionBurstFrameCount.SetValue(24)
# camera.AcquisitionStart()

capture.frames = []

camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)

print('hi')

time.sleep(2)


camera.StopGrabbing()
# camera.AcquisitionStop()

frames = np.array(capture.frames.copy())

np.save('./tools/frames_temp.npy', frames)
