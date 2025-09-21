import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class WebcamWorker(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, device_index=0):
        super().__init__()
        self.cap = cv2.VideoCapture(device_index)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()