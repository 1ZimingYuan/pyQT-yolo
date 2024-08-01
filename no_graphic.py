from fileinput import filename
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
import sys, os, time
import cv2 as cv
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QDir, QUrl, Qt, QFileInfo, QEvent, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QMovie
from PyQt6 import uic
from common import *
from pathlib import Path
from source.ui import test_ui
import torch

file_path = ""
model_path = ""
model = None
cache_file = r"cache\file-model.txt"
model_format = [".pt", ".onnx", ".engine"]
img_format = [".jpg", ".png", ".jpeg", ".bmp", ".dng", ".mpo", ".tif", ".tiff", ".webp", ".pfm"]
video_format = [".mp4", ".avi", ".rmvb", ".mov", ".wmv", ".flv", ".mkv", ".webm", ".asf", ".gif", ".m4v", ".mpeg", ".mpg", ".ts"]
labels = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",9: "traffic light" ,10: "fire hydrant" ,11: "stop sign" ,12: "parking meter" ,13: "bench",14: "bird",15: "cat",16: "dog",17: "horse",18: "sheep",19: "cow",20: "elephant",21: "bear",22: "zebra",23: "giraffe",24: "backpack",25: "umbrella",26: "handbag",27: "tie",28: "suitcase",29: "frisbee",30: "skis",31: "snowboard",32: "sports ball" ,33: "kite",34: "baseball bat" ,35: "baseball glove",36: "skateboard",37: "surfboard",38: "tennis racket",39: "bottle",40: "wine glass",41: "cup",42: "fork",43: "knife",44: "spoon",45: "bowl",46: "banana",47: "apple",48: "sandwich",49: "orange",50: "broccoli",51: "carrot",52: "hot dog",53: "pizza",54: "donut",55: "cake",56: "chair",57: "couch",58: "potted plant",59: "bed",60: "dining table",61: "toilet",62: "tv",63: "laptop",64: "mouse",65: "remote",66: "keyboard",67: "cell phone",68: "microwave",69: "oven",70: "toaster",71: "sink",72: "refrigerator",73: "book",74: "clock",75: "vase",76: "scissors",77: "teddy bear",78: "hair drier",79: "toothbrush"}


class DetectWindow(QMainWindow):
    p = pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__()
        # self.setWindowTitle("AI识别")
        self.ui = test_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.load_cache()
        self.p.connect(lambda string: print(string))
        if not torch.cuda.is_available():
            self.ui.radioButton_gpu.setEnabled(False)

    def load_cache(self):
        """
        加载缓存,文件路径和模型路径
        """
        global file_path, model_path
        with open(cache_file, "r") as f:
            cache = [line.strip("\n") for line in f.readlines()]
            for line in cache:
                if Path(line).suffix in img_format:
                    file_path = line
                elif Path(line).suffix in model_format:
                    model_path = line

    def update_cache(self):
        """
        更新缓存
        """
        global file_path, model_path
        with open(cache_file, "w") as f:
            f.write(file_path + "\n")
            f.write(model_path + "\n")

    def on_psbtn_file_pressed(self):
        """
        获取图片
        """
        global file_path, model_path
        fileName = ""
        curPath = QDir.currentPath() if file_path == "" else str(Path(file_path).parent) if Path(file_path).exists() else QDir.currentPath()
        title = r"获取图片"
        filt = r"图片 *" + " *".join(img_format)
        fileName,flt = QFileDialog.getOpenFileName(self,title,curPath,filt) # 打开文件夹，获取文件名
        if fileName == "":
            return
        # print(fileName)
        fileInfo = QFileInfo(fileName) # 获取文件信息
        file_path = fileInfo.filePath() # 获取文件名
        # if Path(file_path).suffix in img_format:
        self.ui.label_file.setText("文件： "+file_path)
        if self.ui.label_model.text() and self.ui.label_file.text():
            self.ui.psbtn_detect.setEnabled(True)
        self.media_display()

    def on_psbtn_model_pressed(self):
        """
        获取模型
        """
        global model_path, file_path, model
        curPath = QDir.currentPath() if model_path == "" else str(Path(model_path).parent) if Path(model_path).exists() else QDir.currentPath()
        title = r"获取模型"
        filt = r"模型 *" + " *".join(model_format)
        fileName, flt = QFileDialog.getOpenFileName(self, title, curPath, filt) # 打开文件夹，获取文件名
        if fileName == "":
            return
        fileInfo = QFileInfo(fileName) # 获取模型信息
        model_path = fileInfo.filePath() # 获取模型名
        self.ui.label_model.setText("模型： "+model_path)
        model = YOLO(model_path) # 加载模型
        if self.ui.label_model.text() and self.ui.label_file.text():
            self.ui.psbtn_detect.setEnabled(True)
        # detect(filePath) # 调用检测函数

    def on_psbtn_detect_pressed(self):
        """
        开始检测
        """
        global model_path, file_path, model
        self.update_cache() # 更新缓存, 保存文件路径和模型路径
        visual_args = self.get_visual_args() # 获取检测参数
        detect_args = self.get_detect_args() # 获取检测参数
        model.to("cpu" if detect_args["device"] == "cpu" else "cuda:0")
        del detect_args["device"]
        img, cls_tensor,infer_time = detect(file_path, model, detect_args, visual_args) # 调用检测函数
        self.media_detect_display(img) # 显示检测结果
        cls_np = cls_tensor.cpu().numpy()
        temp = {}
        rst_txt = ""
        for i in cls_np:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] += 1
        for key, value in temp.items():
            rst_txt += labels[key] + ": " + str(value) + "\n"
        self.ui.label_rst.setText(rst_txt)  # 显示结果
        self.ui.label_sum.setText(str(cls_np.shape[0]))
        self.ui.label_time.setText(str(int(infer_time)))
    
    def on_radioButton_cpu_toggled(self, checked):
        """
        cpu时取消半精度
        """
        self.ui.checkBox_half.setEnabled(not checked)
        
    def on_checkBox_box_stateChanged(self, state):
        """
        复选框激活
        """
        self.ui.checkBox_lb.setEnabled(True if state == 2 else False)
        if self.ui.checkBox_lb.isEnabled() and self.ui.checkBox_lb.isChecked():
            self.ui.checkBox_conf.setEnabled(True)
        else:
            self.ui.checkBox_conf.setEnabled(False)

    def on_checkBox_lb_stateChanged(self, state):
        """
        复选框激活
        """
        self.ui.checkBox_conf.setEnabled(True if state == 2 else False)

    def media_display(self):
        """
        将原始图片显示到媒体区
        """
        global file_path
        self.media_w = self.ui.media_area.width()
        self.media_h = self.ui.media_area.height()
        h, w, _ = cv.imread(file_path).shape
        if self.media_h/self.media_w>h/w:
            pixmap = QPixmap(file_path).scaled(self.media_w, int(h*self.media_w/w))#, Qt.AspectRatioMode.KeepAspectRatio
        else:
            pixmap = QPixmap(file_path).scaled(int(w*self.media_h/h),  self.media_h)#, Qt.AspectRatioMode.KeepAspectRatio
        self.ui.media_area.setPixmap(pixmap)  # 显示图片

    def media_detect_display(self, img: np.ndarray):
        """
        将检测的图像显示到媒体区
        """
        height, width, _ = img.shape  
        bytes_per_line = 3 * width  # 假设是 RGB 图片  
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()  # 转换为 Qt 格式
        self.media_w = self.ui.media_area.width()
        self.media_h = self.ui.media_area.height()
        if self.media_h/self.media_w>height/width:
            pixmap = QPixmap.fromImage(qimage).scaled(self.media_w, int(height*self.media_w/width))#, Qt.AspectRatioMode.KeepAspectRatio
        else:
            pixmap = QPixmap.fromImage(qimage).scaled(int(width*self.media_h/height), self.media_h)#, Qt.AspectRatioMode.KeepAspectRatio
        self.ui.media_area.setPixmap(pixmap)  # 显示图片

    def get_detect_args(self):
        """
        获取检测参数
        """
        detect_args = {}
        detect_args["conf"] = float(self.ui.doubleSpinBox_conf.text())
        detect_args["iou"] = float(self.ui.doubleSpinBox_iou.text())
        detect_args["max_det"] = int(self.ui.spinBox_maxbox.text())
        if self.ui.checkBox_half.isChecked():
            detect_args["half"] = True
        if self.ui.checkBox__classagnostic.isChecked():
            detect_args["agnostic_nms"] = True
        detect_args["device"] = "0" if self.ui.radioButton_gpu.isChecked() else "cpu"
        return detect_args

    def get_visual_args(self):
        """
        获取可视化参数
        """
        visual_args = {}
        if not self.ui.checkBox_lb.isChecked():
            visual_args["labels"] = False
        if not self.ui.checkBox_box.isChecked():
            visual_args["boxes"] = False
        if not self.ui.checkBox_conf.isChecked():
            visual_args["conf"] = False
        visual_args["line_width"] = int(float(self.ui.doubleSpinBox_line_width.text()))
        visual_args["font_size"] = float(self.ui.doubleSpinBox_font_size.text())
        return visual_args

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectWindow()
    window.show()
    sys.exit(app.exec())