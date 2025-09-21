import cv2
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal
from src.filters.manager import apply_filters

class WebcamWorker(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, camera_index=0, pig_state=None):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.pig_state = pig_state

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for selfie-view
            frame = cv2.flip(frame, 1)

            # Run Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Apply filters depending on pig level
            filtered = apply_filters(frame, results, self.pig_state.level)

            self.frame_ready.emit(filtered)

        cap.release()
        holistic.close()

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()