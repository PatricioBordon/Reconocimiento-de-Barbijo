from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2

class videoCapture (qtc.QThread):  
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0) #puedo establecer mas camaras

        while self.run_flag:
            ret , frame = cap.read() #formato bgr

            if ret == True:
                self.change_pixmap_signal.emit(frame)

        cap.release()

    def stop(self):
        self.run_flag = False
        self.wait()

class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        #gui
        self.setWindowIcon(qtg.QIcon('./imagenes/IconApp.png'))
        self.setWindowTitle('( ͡👁 ͜ʖ ͡👁)')
        self.setGeometry(100,100,700,600)
        #widgets
        label = qtw.QLabel('<h2>Face Mask Recognition</h2>')

        label.setAlignment(qtc.Qt.AlignCenter)
        label.setStyleSheet("background-color: #001624;"
        "color: rgb(255, 255, 255);"
        "font: bold italic 20pt ;")
        
        self.cameraButton = qtw.QPushButton('Click to Open Camera', clicked=self.cameraButtonClick, checkable = True)
        self.cameraButton.setStyleSheet("background-color: rgb(0, 255, 0);"
        "color: rgb(0, 0, 0);"
        "font: bold italic 10pt ;")

        
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(700,600)
        self.img.fill(qtg.QColor('grey'))
        self.screen.setPixmap(self.img)
        

        self.screen = qtw.QLabel('imagen',self)
        self.screen.setPixmap(qtg.QPixmap('./imagenes/fondo.jpg'))
        self.screen.setScaledContents(False)
        self.screen.setAlignment(qtc.Qt.AlignCenter)
        
        self.setStyleSheet("background-color: #001624;")

        #Layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.screen)
        layout.addWidget(self.cameraButton)
        

        self.setLayout(layout)

        self.show()


    def cameraButtonClick(self):
        print('clicked')
        status = self.cameraButton.isChecked()
        if status == True:
            self.cameraButton.setText('Click to Close Camera')
            self.cameraButton.setStyleSheet("background-color: rgb(255, 0, 0);"
            "color: rgb(0, 0, 0);"
            "font: bold italic 10pt ;")

            self.capture = videoCapture()
            self.capture.change_pixmap_signal.connect(self.updateImage)
            self.capture.start()
            



        elif status == False:
            self.cameraButton.setText('Click to Open Camera')
            self.cameraButton.setStyleSheet("background-color: rgb(0, 255, 0);"
            "color: rgb(0, 0, 0);"
            "font: bold italic 10pt ;")

            self.capture.stop()
    
    @qtc.pyqtSlot(np.ndarray)
    def updateImage(self,image_array):
        rgb_img = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB) #BGR A RGB
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        #convertir a QtImage
        convertedImage = qtg.QImage(rgb_img.data, w, h, bytes_per_line, qtg.QImage.Format_RGB888)
        scaledImage = convertedImage.scaled(700,600,qtc.Qt.KeepAspectRatio)
        qt_img = qtg.QPixmap.fromImage(scaledImage)

        #Update to screen
        self.screen.setPixmap(qt_img)


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec())