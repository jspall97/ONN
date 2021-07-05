from pypylon import pylon
import time
import cv2
import queue

# connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
pylon.FeaturePersistence.Load("./tools/pylon_settings.pfs", camera.GetNodeMap())

# if available choose a format that is
# computationally inexpensive to convert to your desired output format

# or the YUV formats

class CaptureConvert(pylon.ImageEventHandler):
    def __init__(self):
        super().__init__()

        self.video_queue = queue.Queue(maxsize=1000)

    def OnImageGrabbed(self, cam, grab_result):
        if grab_result.GrabSucceeded():
            image = grab_result.GetArray()
            try:
                self.video_queue.put_nowait(image)
            except queue.Full:
                # if queue depth > 2 your display thread is too slow
                # try limiting the camera framerate or use faster display framework
                print("consumer thread too slow frame:", grab_result.FrameNumber)


# register the background handler
capture = CaptureConvert()

camera.RegisterImageEventHandler(capture, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)

# start grabbing using background pylon thread
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByInstantCamera)

start_time = time.time()
counter = 0

for i in range(500):
    # Access the image data
    img = capture.video_queue.get(timeout=5000)

    print(img.mean())

for i in range(500):
    # Access the image data
    img = capture.video_queue.get(timeout=5000)

    print(img.mean())


# Releasing the resource
camera.StopGrabbing()
camera.Close()
