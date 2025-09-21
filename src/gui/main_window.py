import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

from src.state.pig_state import PigLevelState
from src.gui.widgets import MeterWidget

class MainWindow(QMainWindow):
    def __init__(self, state):
        super().__init__()

        with open("assets/styles/main.qss", "r") as f:
            self.setStyleSheet(f.read())

        self.setWindowTitle("Curse of Oink 🐷")
        self.resize(1000, 600)

        # Track pig level
        self.state = state

        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: white;")
        self.setCentralWidget(central_widget)

        # Horizontal layout → left meter + right content
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # --- Left pig level barometer
        meter_imgs = []
        for i in range(6):  # levels 0..5
            path = os.path.join("assets", "barometer", f"level_{i}.png")
            meter_imgs.append(QPixmap(path))
        self.meter_widget = MeterWidget(meter_imgs)
        self.meter_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.meter_widget)

        # --- Right side (video + button)
        right_panel = QVBoxLayout()

        self.video_label = QLabel("Webcam will appear here")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.video_label)

        self.pigify_button = QPushButton("Pigify more!")
        self.pigify_button.setObjectName("pigifyButton")
        self.pigify_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pigify_button.adjustSize()
        self.pigify_button.clicked.connect(self.on_pigify_clicked)
        right_panel.addWidget(self.pigify_button, alignment=Qt.AlignHCenter)

        layout.addLayout(right_panel)

        # --- Banner 
        self.game_over_banner = QLabel("GAME OVER", self)
        self.game_over_banner.setObjectName("gameOverBanner")
        self.game_over_banner.setAlignment(Qt.AlignCenter)
        self.game_over_banner.hide()
        self.resizeEvent = self.on_resize  # Update width when window resizes

        # Timer for blinking
        self.banner_timer = QTimer()
        self.banner_timer.setInterval(500)  # in ms
        self.banner_timer.timeout.connect(self.toggle_banner)

    def on_pigify_clicked(self):
        self.state.increase()
        self.meter_widget.set_level(self.state.level)
        print(f"Pig level: {self.state.level}")

        # Check if level 5 reached
        if self.state.level >= 5:
            self.pigify_button.setDisabled(True)  # disable button
            self.banner_timer.start()

    def set_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale pixmap to label while keeping aspect ratio
        pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def set_webcam(self, webcam_worker):
        """Register webcam worker so we can stop it when closing."""
        self.webcam_worker = webcam_worker

    def toggle_banner(self):
        if self.game_over_banner.isVisible():
            self.game_over_banner.hide()
        else:
            self.game_over_banner.show()

    def on_resize(self, event):
        self.game_over_banner.setGeometry(
            0,                   # x
            self.height() // 2,  # y
            self.width(),        # full width
            self.height() // 5   # height
        )
        event.accept()

    def closeEvent(self, event):
        """Handle window close event."""
        if hasattr(self, "webcam_worker") and self.webcam_worker is not None:
            self.webcam_worker.stop()
        event.accept()
