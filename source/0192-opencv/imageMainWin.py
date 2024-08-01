from PyQt6.QtWidgets import QWidget, QApplication,QFileDialog
from ImageWin import Ui_ImageForm
from PyQt6.QtCore import QDir
import cv2 as cv
from PyQt6.QtGui import QPixmap,QImage
import sys

class QmyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__ui = Ui_ImageForm()
        self.__ui.setupUi(self)
        self.fileName = ""

    def on_pushButton_OpenImage_pressed(self):
        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "图片文件(*.png *.jpg);;所有文件(*.*)"
        self.fileName,flt = QFileDialog.getOpenFileName(self,title,curPath,filt)
        if self.fileName == "":
            return
        pixmap = QPixmap(self.fileName)
        if(pixmap.width()>400):
            ratio = pixmap.width()/400
            pixmap.setDevicePixelRatio(ratio)
        self.__ui.label_originImage.setPixmap(pixmap)
        dim = min(pixmap.width(),pixmap.height(),400)
        self.__ui.blurSlider.setMaximum(dim)

    def on_pushButton_BlurImage_pressed(self):
        cvimg = cv.imread(self.fileName)
        cvimg = cv.blur(cvimg,(5,5))
        print(cvimg.data)
        #CV2:cvimg.shape图像属性包括行数，列数和通道数，图像数据类型，像素数等
        #QImage:QImage(bytes,width,height,format)
        img = QImage(cvimg.data,cvimg.shape[1],cvimg.shape[0],QImage.Format.Format_BGR888) 
        pixmap = QPixmap.fromImage(img)
        if(pixmap.width()>400):
            ratio = pixmap.width()/400
            pixmap.setDevicePixelRatio(ratio)
        self.__ui.label_blueImage.setPixmap(pixmap)
        self.__ui.blurSlider.setEnabled(True)
        self.__ui.blurSlider.setValue(5)
        
    def on_blurSlider_valueChanged(self,value):
        cvimg = cv.imread(self.fileName)
        cvimg = cv.blur(cvimg,(value,value))
        #CV2:cvimg.shape图像属性包括行数，列数和通道数，图像数据类型，像素数等
        #QImage:QImage(bytes,width,height,format)
        img = QImage(cvimg.data,cvimg.shape[1],cvimg.shape[0],QImage.Format.Format_BGR888) 
        pixmap = QPixmap.fromImage(img)
        if(pixmap.width()>400):
            ratio = pixmap.width()/400
            pixmap.setDevicePixelRatio(ratio)
        self.__ui.label_blueImage.setPixmap(pixmap) 


if  __name__ == "__main__":
   app = QApplication(sys.argv)   #创建App，用QApplication类
   myWidget=QmyWidget()
   myWidget.show()
   sys.exit(app.exec())     