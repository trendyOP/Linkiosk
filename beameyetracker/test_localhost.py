import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로컬호스트 뷰어")
        self.setGeometry(100, 100, 1200, 800)

        # 웹 뷰 생성
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("http://localhost:3000"))
        self.setCentralWidget(self.browser)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 