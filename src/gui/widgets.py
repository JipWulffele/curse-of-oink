import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class VideoLabel(QLabel):
    def __init__(self):
        pass

    def set_frame(self, frame):
        """frame is BGR numpy array"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

"""
class MeterWidget(QLabel):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.images = images  # list of QPixmaps
        self.setPixmap(self.images[0])

    def set_level(self, level):
        if 0 <= level < len(self.images):
            self.setPixmap(self.images[level])
"""

class MeterWidget(QLabel):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.images = images  # list of QPixmaps
        self.current_level = 0
        self.setAlignment(Qt.AlignCenter)
        self.setPixmap(self.images[0])
        self.setScaledContents(False)  # we’ll scale manually

    def set_level(self, level):
        if 0 <= level < len(self.images):
            self.current_level = level
            self.update_pixmap()

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)

    def update_pixmap(self):
        """Scale the current level pixmap to the widget size while keeping aspect ratio."""
        pixmap = self.images[self.current_level]
        scaled_pixmap = pixmap.scaled(
            self.width(),
            self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)