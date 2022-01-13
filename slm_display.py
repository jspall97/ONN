import wx
import threading
import time

EVT_NEW_IMAGE = wx.PyEventBinder(wx.NewEventType(), 0)


class ImageEvent(wx.PyCommandEvent):
    def __init__(self, eventType=EVT_NEW_IMAGE.evtType[0], id=0):
        wx.PyCommandEvent.__init__(self, eventType, id)
        self.img = None
        self.color = False
        self.oldImageLock = None
        self.eventLock = None


class SLMframe(wx.Frame):
    """Frame used to display full screen image."""

    def __init__(self, x0, resX, resY, name, isImageLock=True):
        style = wx.BORDER_NONE | wx.STAY_ON_TOP
        self.isImageLock = isImageLock
        # Set the frame to the position and size of the target monitor
        super().__init__(None, -1, f"{name}", pos=(x0, 0), size=(resX, resY), style=style)
        self.img = wx.Image(2, 2)
        self.bmp = self.img.ConvertToBitmap()
        self.clientSize = self.GetClientSize()
        self.img1 = wx.Image(2, 2)
        self.bmp1 = self.img1.ConvertToBitmap()
        self.clientSize1 = self.GetClientSize()
        # Update the image upon receiving an event EVT_NEW_IMAGE
        self.Bind(EVT_NEW_IMAGE, self.UpdateImage)
        # Set full screen
        # self.ShowFullScreen(not self.IsFullScreen(), wx.FULLSCREEN_ALL)
        self.SetFocus()

    def InitBuffer(self):
        self.bmp = self.img.Scale(self.clientSize[0], self.clientSize[1]).ConvertToBitmap()
        dc = wx.ClientDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def UpdateImage(self, event):
        self.eventLock = event.eventLock
        self.img = event.img
        self.InitBuffer()
        self.ReleaseEventLock()

    def ReleaseEventLock(self):
        if self.eventLock:
            if self.eventLock.locked():
                self.eventLock.release()


class SLMdisplay:
    """Interface for sending images to the display frame."""

    def __init__(self, x0, resX, resY, name, x0_1, resX1, resY1, name1, isImageLock=True):
        self.isImageLock = isImageLock
        self._x0 = x0
        self._resX = resX
        self._resY = resY
        self._name = name
        self._x0_1 = x0_1
        self._resX1 = resX1
        self._resY1 = resY1
        self._name1 = name1
        # Create the thread in which the window app will run
        # It needs its thread to continuously refresh the window
        self.vt = videoThread(self)
        self.eventLock = threading.Lock()
        if self.isImageLock:
            self.eventLock = threading.Lock()

    def updateArray(self, array, array1):
        """
        Update the SLM monitor with the supplied array.
        Note that the array is not the same size as the SLM resolution,
        the image will be deformed to fit the screen.
        """
        # create a wx.Image from the arrays
        h, w = array.shape[0], array.shape[1]
        data = array.tobytes()
        img = wx.ImageFromBuffer(width=w, height=h, dataBuffer=data)

        h1, w1 = array1.shape[0], array1.shape[1]
        data1 = array1.tobytes()
        img1 = wx.ImageFromBuffer(width=w1, height=h1, dataBuffer=data1)

        # Create the event
        event = ImageEvent()
        event.img = img
        event.eventLock = self.eventLock

        # Wait for the lock to be released (if isImageLock = True)
        # to be sure that the previous image has been displayed
        # before displaying the next one - it avoids skipping images
        if self.isImageLock:
            event.eventLock.acquire()

        # time.sleep(0.1)

        self.vt.frame.AddPendingEvent(event)

        # Create the event
        event = ImageEvent()
        event.img = img1
        event.eventLock = self.eventLock

        # Wait for the lock to be released (if isImageLock = True)
        # to be sure that the previous image has been displayed
        # before displaying the next one - it avoids skipping images
        if self.isImageLock:
            event.eventLock.acquire()

        self.vt.frame1.AddPendingEvent(event)


class videoThread(threading.Thread):
    """Run the MainLoop as a thread. Access the frame with self.frame."""

    def __init__(self, parent, autoStart=True):
        super().__init__()
        self.parent = parent
        # Set as deamon so that it does not prevent the main program from exiting
        self.setDaemon(True)
        self.start_orig = self.start
        self.start = self.start_local
        self.frame = None  # to be defined in self.run
        self.frame1 = None
        self.lock = threading.Lock()
        self.lock.acquire()  # lock until variables are set
        if autoStart:
            self.start()  # automatically start thread on init

    def run(self):
        self.app = wx.App()
        frame = SLMframe(self.parent._x0, self.parent._resX, self.parent._resY,
                         self.parent._name)
        frame.Show(True)
        frame1 = SLMframe(self.parent._x0_1, self.parent._resX1, self.parent._resY1,
                         self.parent._name1)
        frame1.Show(True)
        self.frame = frame
        self.frame1 = frame1
        self.lock.release()
        self.app.MainLoop()

    def start_local(self):
        self.start_orig()
        # Use lock to wait for the functions to get defined
        self.lock.acquire()
