import sys
from PyQt5.QtWidgets import QApplication
from GUI.MainWidget import MainWidget


def main():
    app = QApplication(sys.argv)
    GUI = MainWidget()
    GUI.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
