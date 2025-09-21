from PyQt5.QtCore import QObject, pyqtSignal

class PigLevelState(QObject):
    level_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.level = 0 # pig level 0 (nothing) - 6 = bacon

    def increase(self):
        if self.level < 5:
            self.level += 1
            self.level_changed.emit(self.level)