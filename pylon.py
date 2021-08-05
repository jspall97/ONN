from pypylon import pylon
from pypylon import genicam
import time
import numpy as np
import sys
import threading

dims = sys.argv[1:4]
n = int(dims[0])
m = int(dims[1])

# x_edge_indxs = np.load('./tools/x_edge_indxs.npy')
# y_center_indxs = np.load('./tools/y_center_indxs.npy')
#
# area_height = 7
# half_height = area_height // 2


# def find_spot_ampls(arr):
#     def spot_s_200(i):
#         y_center_i = y_center_indxs[i]
#         return np.s_[x_edge_indxs[2 * i]:x_edge_indxs[2 * i + 1],
#                y_center_i - half_height:y_center_i + half_height + 1]
#
#     spots_dict = {}
#
#     for spot_num in range(m):
#         spot = arr.T[spot_s_200(spot_num)]
#
#         mask = spot < 3
#         spot -= 2
#         spot[mask] = 0
#
#         spots_dict[spot_num] = spot
#
#     spot_powers = np.array([spots_dict[i].mean() for i in range(m)])
#
#     spot_ampls = np.sqrt(spot_powers)
#
#     spot_ampls = np.flip(spot_ampls)
#
#     return np.array(spot_ampls)
# #
#
def run_camera(conn, action='init'):

    try:
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1)
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        camera.Open()

        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        print()

        pylon.FeaturePersistence.Load("./tools/pylon_settings.pfs", camera.GetNodeMap())

        def pylon_collect_frames(max_frames, batch):
            global all_frames_arr, ts_arr, frame_count

            frame_count = 0
            # all_ampls = []
            all_frames = []
            # timestamps = []

            t0 = time.time()

            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

            while camera.IsGrabbing():

                # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                # Image grabbed successfully?
                if grabResult.GrabSucceeded():

                    frame_count += 1
                    image = grabResult.Array
                    # ampls = find_spot_ampls(image)

                    all_frames.append(image)
                    # all_ampls.append(ampls)

                    # t1 = grabResult.ChunkTimestamp.Value

                    # timestamps.append(t1)

                    imageWindow.SetImage(grabResult)
                    imageWindow.Show()

                else:
                    print("Error: ", grabResult.ErrorCode)

                grabResult.Release()

                if frame_count == max_frames:
                    camera.StopGrabbing()

                    # ts_arr = np.array(timestamps)
                    # ts_arr -= ts_arr[0]
                    # ts_arr = ts_arr / 1e9
                    # ts_arr += t0

                    all_frames_arr = np.array(all_frames)
                    # all_ampls_arr = np.array(all_ampls)

                    if action == 'init':
                        # np.save('./tools/init_captures/ampls/batch_{}.npy'.format(batch), all_ampls_arr)
                        np.save('./tools/init_captures/frames/batch_{}.npy'.format(batch), all_frames_arr)
                        # np.save('./tools/init_captures/timestamps/batch_{}.npy'.format(batch), ts_arr)

                    elif action == 'train':

                        # np.save('./MNIST/pylon_captures/ampls/batch_{}.npy'.format(batch), all_ampls_arr)
                        np.save('./MNIST/pylon_captures/frames/batch_{}.npy'.format(batch), all_frames_arr)
                        # np.save('./MNIST/pylon_captures/timestamps/batch_{}.npy'.format(batch), ts_arr)

                    print('pylon finished batch {}'.format(batch))

        while True:

            while not conn.poll():
                pass
            batch, num_frames = conn.recv()
            if batch == 0:
                pass  # initial frames, ignore
            if batch == 1:
                break  # finish signal, stop camera
            else:
                pylon_collect_frames(num_frames, batch-2)
                conn.send(1)

        # camera has to be closed manually
        camera.Close()
        # imageWindow has to be closed manually
        imageWindow.Close()

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.")
        print(e)


def view_camera():

    try:
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1)
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()

        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        print()

        pylon.FeaturePersistence.Load("./tools/pylon_settings.pfs", camera.GetNodeMap())
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        while camera.IsGrabbing():

            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                imageWindow.SetImage(grabResult)
                imageWindow.Show()
                sys.stdout.write("\r\rMax value: {}".format(grabResult.Array.max()))
                sys.stdout.flush()

            else:
                print("Error: ", grabResult.ErrorCode)

            grabResult.Release()
            time.sleep(0.05)

            if not imageWindow.IsVisible():
                camera.StopGrabbing()
                print()

        # camera has to be closed manually
        camera.Close()
        # imageWindow has to be closed manually
        imageWindow.Close()

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.")
        print(e)


class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        view_camera()


class Camera():
    def __init__(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        pylon.FeaturePersistence.Load("C:/Users/spall/PycharmProjects/ONN/tools/pylon_settings.pfs",
                                      self.camera.GetNodeMap())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        self.arr = grabResult.Array
        # self.ampls = np.empty(m)

    def capture(self):
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        self.arr = grabResult.Array
        # self.ampls = find_spot_ampls(self.arr)

    def close(self):
        self.camera.Close()
