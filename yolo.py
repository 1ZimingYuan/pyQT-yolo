from fileinput import filename
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
import sys
import cv2 as cv
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QDir, QUrl, Qt, QFileInfo, QEvent
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6 import uic
from common import *
from pathlib import Path
from source.ui import test_ui

file_path = ""
model_path = ""
model_format = [".pt", ".onnx", ".engine"]
img_format = [".jpg", ".png", ".jpeg", ".bmp", ".dng", ".mpo", ".tif", ".tiff", ".webp", ".pfm"]
video_format = [".mp4", ".avi", ".rmvb", ".mov", ".wmv", ".flv", ".mkv", ".webm", ".asf", ".gif", ".m4v", ".mpeg", ".mpg", ".ts"]


def on_pushButton_3_pressed(widget):
    """
    获取图片
    """
    global file_path, model_path
    fileName = ""
    curPath = QDir.currentPath()
    title = r"获取图片"
    filt = r"图片 *" + " *".join(img_format)
    fileName,flt = QFileDialog.getOpenFileName(widget,title,curPath,filt) # 打开文件夹，获取文件名
    if fileName == "":
        return
    print(fileName)
    fileInfo = QFileInfo(fileName) # 获取文件信息
    file_path = fileInfo.filePath() # 获取文件名
    f_ = Path(file_path)
    m_ = Path(model_path)
    if f_.suffix in img_format and m_.suffix in model_format:
        widget.pushButton.setEnabled(True)
    # detect(filePath) # 调用检测函数

def on_pushButton_2_pressed(widget):
    """
    获取模型
    """
    global model_path, file_path
    curPath = QDir.currentPath()
    title = r"获取模型"
    filt = r"模型 *" + " *".join(model_format)
    fileName, flt = QFileDialog.getOpenFileName(widget, title, curPath, filt) # 打开文件夹，获取文件名
    if fileName == "":
        return
    fileInfo = QFileInfo(fileName) # 获取模型信息
    model_path = fileInfo.filePath() # 获取模型名
    f_ = Path(file_path)
    m_ = Path(model_path)
    if f_.suffix in img_format and m_.suffix in model_format:
        widget.pushButton.setEnabled(True)
    # detect(filePath) # 调用检测函数

def on_pushButton_pressed(widget):
    """
    开始检测
    """
    global model_path, file_path
    model = YOLO(model_path) # 加载模型
    img:np.ndarray = detect(file_path, model) # 调用检测函数

    height, width, channels = img.shape  
    bytes_per_line = 3 * width  # 假设是 RGB 图片  
    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()  # 转换为 Qt 格式
    label_w = widget.label.width()
    label_h = widget.label.height()
    if label_h/label_w>height/width:
        pixmap = QPixmap.fromImage(qimage).scaled(label_w, int(height*label_w/width))#, Qt.AspectRatioMode.KeepAspectRatio
    else:
        pixmap = QPixmap.fromImage(qimage).scaled(int(width*label_h/height), label_h)#, Qt.AspectRatioMode.KeepAspectRatio
    widget.label.setPixmap(pixmap)  # 显示图片


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = uic.loadUi(r'source\ui\test.ui')
    print(window)
    # print(dir(window))
    # window.pushButton_3.clicked.connect(lambda: on_pushButton_3_pressed(window))
    # window.pushButton_2.clicked.connect(lambda: on_pushButton_2_pressed(window))
    # window.pushButton.clicked.connect(lambda: on_pushButton_pressed(window))
    window.show()
    sys.exit(app.exec())

