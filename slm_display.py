import wx
import threading
import cupy

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

    def __init__(self, isImageLock=True):
        self.isImageLock = isImageLock
        # Create the frame
        self._x0 = 0
        self._y0 = 0
        self._resX = 1920
        self._resY = 1080
        # Set the frame to the position and size of the target monitor
        wx.Frame.__init__(self, None, -1, 'SLM window', pos=(self._x0, self._y0), size=(self._resX, self._resY))
        self.img = wx.Image(2, 2)
        self.bmp = self.img.ConvertToBitmap()
        self.clientSize = self.GetClientSize()
        # Update the image upon receiving an event EVT_NEW_IMAGE
        self.Bind(EVT_NEW_IMAGE, self.UpdateImage)
        # Set full screen
        self.ShowFullScreen(not self.IsFullScreen(), wx.FULLSCREEN_ALL)
        self.SetFocus()

    def InitBuffer(self):
        self.clientSize = self.GetClientSize()
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

    def __init__(self, monitor=0, isImageLock=False):
        self.isImageLock = isImageLock
        self.monitor = monitor
        # Create the thread in which the window app will run
        # It needs its thread to continuously refresh the window
        self.vt = videoThread(self)
        self.eventLock = threading.Lock()
        if (self.isImageLock):
            self.eventLock = threading.Lock()

    def getSize(self):
        return self.vt.frame._resX, self.vt.frame._resY

    def updateArray(self, array):
        """
        Update the SLM monitor with the supplied array.
        Note that the array is not the same size as the SLM resolution,
        the image will be deformed to fit the screen.
        """
        # create a wx.Image from the array
        h, w = array.shape[0], array.shape[1]
        data = array.tobytes()
        img = wx.ImageFromBuffer(width=w, height=h, dataBuffer=data)
        # Create the event
        event = ImageEvent()
        event.img = img
        event.eventLock = self.eventLock

        # Wait for the lock to be released (if isImageLock = True)
        # to be sure that the previous image has been displayed
        # before displaying the next one - it avoids skipping images
        if (self.isImageLock):
            event.eventLock.acquire()

        self.vt.frame.AddPendingEvent(event)

    def close(self):
        raise KeyboardInterrupt
        # self.vt.frame.Close()
        # sys.exit()


class videoThread(threading.Thread):
    """Run the MainLoop as a thread. Access the frame with self.frame."""

    def __init__(self, parent, autoStart=True):
        threading.Thread.__init__(self)
        self.parent = parent
        # Set as deamon so that it does not prevent the main program from exiting
        self.setDaemon(1)
        self.start_orig = self.start
        self.start = self.start_local
        self.frame = None  # to be defined in self.run
        self.lock = threading.Lock()
        self.lock.acquire()  # lock until variables are set
        if autoStart:
            self.start()  # automatically start thread on init

    def run(self):
        app = wx.App()
        frame = SLMframe(isImageLock=self.parent.isImageLock)
        frame.Show(True)
        self.frame = frame
        self.lock.release()
        # Start GUI main loop
        app.MainLoop()

    def start_local(self):
        self.start_orig()
        # Use lock to wait for the functions to get defined
        self.lock.acquire()
