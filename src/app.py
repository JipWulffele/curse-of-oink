import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.camera.webcam import WebcamWorker
from src.state.pig_state import PigLevelState

def main():
    app = QApplication(sys.argv)
    state = PigLevelState()
    win = MainWindow(state)

    webcam = WebcamWorker(0, state)
    win.set_webcam(webcam)  # give reference to MainWindow

    def on_frame(frame):
        win.set_frame(frame) # For now, just show raw frame

    webcam.frame_ready.connect(on_frame)

    win.show()
    webcam.start()
    app.exec_()  # when window closes, closeEvent stops webcam

if __name__ == "__main__":
    main()