import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap

class VideoLabel(QLabel):
    def __init__(self):
        pass

    def set_frame(self, frame):
        """frame is BGR numpy array"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

class MeterWidget(QLabel):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.images = images  # list of QPixmaps
        self.setPixmap(self.images[0])

    def set_level(self, level):
        if 0 <= level < len(self.images):
            self.setPixmap(self.images[level])