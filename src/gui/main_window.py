import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from src.state.pig_state import PigLevelState
from src.gui.widgets import MeterWidget


class MainWindow(QMainWindow):
    def __init__(self, state):
        super().__init__()

        self.setWindowTitle("Curse of Oink 🐷")
        self.resize(1000, 600)

        # Track pig level
        self.state = state

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Horizontal layout → left meter + right content
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # --- Left meter
        meter_imgs = []
        for i in range(6):  # levels 0..5
            path = os.path.join("assets", "barometer", f"level_{i}.png")
            meter_imgs.append(QPixmap(path))
        self.meter_widget = MeterWidget(meter_imgs)
        layout.addWidget(self.meter_widget)

        # --- Right side (video + button)
        right_panel = QVBoxLayout()

        self.video_label = QLabel("Webcam will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.video_label)

        self.pigify_button = QPushButton("Pigify more!")
        self.pigify_button.clicked.connect(self.on_pigify_clicked)
        right_panel.addWidget(self.pigify_button)

        layout.addLayout(right_panel)

    def on_pigify_clicked(self):
        self.state.increase()
        self.meter_widget.set_level(self.state.level)
        print(f"Pig level: {self.state.level}")

    def set_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def set_webcam(self, webcam_worker):
        """Register webcam worker so we can stop it when closing."""
        self.webcam_worker = webcam_worker

    def closeEvent(self, event):
        """Handle window close event."""
        if hasattr(self, "webcam_worker") and self.webcam_worker is not None:
            self.webcam_worker.stop()
        event.accept()
