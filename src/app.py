import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.camera.webcam import WebcamWorker

def main():
    app = QApplication(sys.argv)
    win = MainWindow()

    webcam = WebcamWorker(0)
    win.set_webcam(webcam)  # give reference to MainWindow

    def on_frame(frame):
        win.set_frame(frame) # For now, just show raw frame

    webcam.frame_ready.connect(on_frame)

    win.show()
    webcam.start()
    app.exec_()  # when window closes, closeEvent stops webcam

if __name__ == "__main__":
    main()