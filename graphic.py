from fileinput import filename
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QDialog, QListWidgetItem, QAbstractItemView, QMessageBox
import sys
import cv2 as cv
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QDir, QUrl, Qt, QFileInfo, QEvent, pyqtSignal, QSize, QRectF
from PyQt6.QtGui import QIcon, QImage, QPixmap, QMovie, QPainter, QImageWriter, QPen, QColor
from PyQt6 import uic
from common_graphic import *
from pathlib import Path
from source.ui import graphic_ui, graphic_cls_ui
import torch
import numpy as np
from rec_2 import RecWidget, Graficview

file_path = ""
model_path = ""
save_dir = ""
model = None
cache_file = r"cache\file-cache.txt"
model_format = [".pt", ".onnx", ".engine"]
img_format = [".jpg", ".png", ".jpeg", ".bmp", ".dng", ".mpo", ".tif", ".tiff", ".webp", ".pfm"]
video_format = [".mp4", ".avi", ".rmvb", ".mov", ".wmv", ".flv", ".mkv", ".webm", ".asf", ".gif", ".m4v", ".mpeg", ".mpg", ".ts"]
gt_labels = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",9: "traffic light" ,10: "fire hydrant" ,11: "stop sign" ,12: "parking meter" ,13: "bench",14: "bird",15: "cat",16: "dog",17: "horse",18: "sheep",19: "cow",20: "elephant",21: "bear",22: "zebra",23: "giraffe",24: "backpack",25: "umbrella",26: "handbag",27: "tie",28: "suitcase",29: "frisbee",30: "skis",31: "snowboard",32: "sports ball" ,33: "kite",34: "baseball bat" ,35: "baseball glove",36: "skateboard",37: "surfboard",38: "tennis racket",39: "bottle",40: "wine glass",41: "cup",42: "fork",43: "knife",44: "spoon",45: "bowl",46: "banana",47: "apple",48: "sandwich",49: "orange",50: "broccoli",51: "carrot",52: "hot dog",53: "pizza",54: "donut",55: "cake",56: "chair",57: "couch",58: "potted plant",59: "bed",60: "dining table",61: "toilet",62: "tv",63: "laptop",64: "mouse",65: "remote",66: "keyboard",67: "cell phone",68: "microwave",69: "oven",70: "toaster",71: "sink",72: "refrigerator",73: "book",74: "clock",75: "vase",76: "scissors",77: "teddy bear",78: "hair drier",79: "toothbrush"}
colors = [(211, 44, 31), (172, 194, 217), (86, 174, 87), (178, 153, 110), (168, 255, 4), (137, 69, 133), (212, 255, 255), (252, 252, 129), 
          (56, 128, 4), (239, 180, 53), (12, 6, 247), (55, 120, 191), (5, 255, 166), (31, 99, 87), (12, 181, 119), (255, 7, 137), 
          (255, 99, 233), (67, 5, 65), (255, 178, 208), (173, 144, 13), (104, 50, 227), (133, 14, 4), (64, 253, 20), (246, 104, 142), 
          (118, 253, 168), (1, 70, 0), (65, 253, 254), (12, 23, 147), (165, 0, 85), (173, 3, 222), (174, 255, 110), (255, 8, 232), 
          (255, 253, 1), (1, 101, 252), (249, 115, 6)]

class DetectWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        # self.setWindowTitle("AI识别")
        self.ui = graphic_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.cls_ui = ClassSelector(sorted(gt_labels.values()), self)
        self.load_cache()
        self.scale = 1.
        self.pos_lis = []
        self.cls_lis = []    
        self.prob_lis = []
        self.items:list[QListWidgetItem] = [] # 记录QListWidget中的项
        self.rects:list[RecWidget] = []
        self.labels = [self.ui.lineEdit_cls, self.ui.lineEdit_pro, self.ui.lineEdit_xmin, self.ui.lineEdit_ymin, self.ui.lineEdit_xmax, self.ui.lineEdit_ymax]
        if not torch.cuda.is_available():
            self.ui.radioButton_gpu.setEnabled(False)
        self.init_ui()
    
    def init_ui(self):
        """
        初始化界面
        """
        self.media_area = Graficview(self)# QtWidgets.QLabel(parent=self.centralwidget)
        self.media_area.setObjectName("media_area")
        self.ui.verticalLayout.addWidget(self.media_area)
        # self.ui.verticalLayout.setStretch(0, 1)
        self.ui.verticalLayout.setStretch(3, 10)

    def load_cache(self):
        """
        加载缓存,文件路径和模型路径
        """
        global file_path, model_path, save_dir
        with open(cache_file, "r") as f:
            cache = [line.strip("\n") for line in f.readlines()]
            if len(cache) == 3:
                file_path, model_path, save_dir = cache

    def update_cache(self):
        """
        更新缓存
        """
        global file_path, model_path, save_dir
        with open(cache_file, "w") as f:
            f.write(file_path + "\n")
            f.write(model_path + "\n")
            f.write(save_dir + "\n")

    def on_psbtn_file_pressed(self):
        """
        获取图片
        """
        global file_path
        fileName = ""
        curPath = QDir.currentPath() if file_path == "" else str(Path(file_path).parent) if Path(file_path).exists() else QDir.currentPath()
        title = r"选择文件"
        filt = r"图片 *" + " *".join(img_format)
        fileName,flt = QFileDialog.getOpenFileName(self,title,curPath,filt) # 打开文件夹，获取文件名
        if fileName == "":
            return
        fileInfo = QFileInfo(fileName) # 获取文件信息
        file_path = fileInfo.filePath() # 获取文件名
        self.ui.label_file.setText("文件： "+file_path)
        if self.ui.label_model.text() and self.ui.label_file.text():
            self.ui.psbtn_detect.setEnabled(True) 
        self.update_cache() # 更新缓存, 保存文件路径和模型路径
        self.media_display()

    def on_psbtn_model_pressed(self):
        """
        获取模型
        """
        global model_path, model
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
        self.update_cache() # 更新缓存, 保存文件路径和模型路径

    def on_psbtn_detect_pressed(self):
        """
        开始检测
        """
        global model_path, file_path, model
        detect_args = self.get_detect_args() # 获取检测参数
        model.to("cpu" if detect_args["device"] == "cpu" else "cuda:0")
        del detect_args["device"]
        self.cls_lis, self.prob_lis, self.pos_lis, infer_time = detect(file_path, model, detect_args) # 调用检测函数
        self.ui.pushButton_draw.setEnabled(True)
        self.ui.label_time.setText(str(int(infer_time)))
        self.media_detect_display(self.pos_lis, self.cls_lis, self.prob_lis) # 显示检测结果

    def on_psbtn_export_pressed(self):
        """
        导出结果
        """
        global file_path, save_dir
        # folder_path = QFileDialog.getExistingDirectory(self, "保存图片")  
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", save_dir, "PNG Files (*.png)")
        if file_name:
            self.draw_save(file_path, file_name)
            """# file_path = QDir(folder_path).filePath("image.png")
            w, h = self.media_area.pixmap_item.pixmap().width(), self.media_area.pixmap_item.pixmap().height()
            pixmap = QPixmap(QSize(w, h))  
            painter = QPainter(pixmap)  
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)  
            self.media_area.scene_.render(painter, source=QRectF(0., 0., float(w), float(h)), )  
            painter.end()  
            # 保存图片  
            image_writer = QImageWriter(file_name)  
            image_writer.write(pixmap.toImage()) """ 
            save_dir = str(Path(file_name).parent)
            save_dir = save_dir.replace("\\", "/")
            self.update_cache() # 更新缓存, 保存文件路径和模型路径

    def on_radioButton_cpu_toggled(self, checked):
        """
        cpu时取消半精度
        """
        self.ui.checkBox_half.setEnabled(not checked)
        
    def on_checkBox_box_stateChanged(self, state):
        """
        复选框激活
        """
        self.ui.checkBox_uniform_cls.setEnabled(True if state == 2 else False)
        if state == 2:
            if not self.ui.checkBox_uniform_cls.isChecked():
                self.ui.checkBox_lb.setEnabled(True)
                self.ui.checkBox_conf.setEnabled(True if self.ui.checkBox_lb.isChecked() else False)
            else:
                self.ui.checkBox_lb.setEnabled(False)
                self.ui.checkBox_conf.setEnabled(True)
        else:
            self.ui.checkBox_lb.setEnabled(False)
            self.ui.checkBox_conf.setEnabled(False)

    def on_checkBox_lb_stateChanged(self, state):
        """
        复选框激活
        """
        self.ui.checkBox_conf.setEnabled(True if state == 2 else False)

    def on_checkBox_uniform_cls_stateChanged(self, state):
        """
        复选框激活
        """
        if state == 0:
            self.ui.checkBox_lb.setEnabled(True)
            self.ui.checkBox_conf.setEnabled(True if self.ui.checkBox_lb.isChecked() else False)
        elif state == 2:
            self.ui.checkBox_lb.setEnabled(False)
            self.ui.checkBox_conf.setEnabled(True)
        self.ui.checkBox_lb.setEnabled(False if state == 2 else True)
        self.ui.checkBox_conf.setEnabled(True if state == 2 else False)
        if state == 0: 
            self.ui.checkBox_conf.setEnabled(True if self.ui.checkBox_lb.isChecked() else False)

    def on_pushButton_draw_pressed(self):
        """
        绘制检测结果
        """
        self.media_area.remove_rec()
        self.media_detect_display(self.pos_lis, self.cls_lis, self.prob_lis)
    
    def on_checkBox_sort_stateChanged(self, state):
        """
        排序
        """
        if state == 2:
            self.ui.listWidget_rst.sortItems()
        else:
            self.ui.listWidget_rst.clear()
            self.ui.listWidget_rst.addItems(self.items)
            ...

    def on_listWidget_rst_itemSelectionChanged(self):
        """
        列表项点击
        """
        for i, item in enumerate(self.items):
            if item.isSelected():
                self.rects[i].setPen(QPen(Qt.GlobalColor.red, self.rects[i].size+2))
            else:
                self.rects[i].setPen(QPen(QColor(*self.rects[i].color), self.rects[i].size))

    def on_toolButton_cls_pressed(self):
        """
        选择类别
        """
        self.cls_ui.exec()
        items = []
        for i in range(self.cls_ui.ui.listWidget_tgt.count()):
            items.append(self.cls_ui.ui.listWidget_tgt.item(i).text())
        if len(items)<3 and len(items)>0:
            cls_txt = ",".join(items)
        elif len(items)==0:
            cls_txt = "无类别"
        elif len(items)==self.cls_ui.ui.listWidget_src.count():
            cls_txt = "所有类别"
        else:
            cls_txt = cls_txt = ",".join(items[:2])+", ..."
        self.ui.toolButton_cls.setText(cls_txt)

    def media_display(self):
        """
        将原始图片显示到媒体区
        """
        global file_path
        self.media_w = self.media_area.width()
        self.media_h = self.media_area.height()
        h, w, _ = cv.imread(file_path).shape
        if self.media_h/self.media_w>h/w:
            self.scale = self.media_w/w
            self.media_area.add_img(file_path, self.media_w, int(h*self.media_w/w))
        else:
            self.scale = self.media_h/h 
            self.media_area.add_img(file_path, int(w*self.media_h/h), self.media_h)
        self.ui.pushButton_draw.setEnabled(False)
        
    def media_detect_display(self, pos_lis:list[tuple], 
                             cls_lis:list[str], 
                             prob_lis:list[float]):
        """
        将检测的图像显示到媒体区
        """
        visual_args = self.get_visual_args()
        rst_txt = ""
        obj_count = {}
        obj_color = {}
        temp_id = 0
        self.items = []
        self.rects = []
        self.media_area.remove_rec()
        self.ui.listWidget_rst.clear()
        if visual_args["draw_box"]:
            for obj_id in range(len(pos_lis)):
                lb = gt_labels[cls_lis[obj_id]]
                if lb in visual_args["cls"]:
                    if lb not in obj_color:
                        obj_color[lb] = colors[temp_id] if temp_id < len(colors) else (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        temp_id += 1
                    if lb not in obj_count:
                        obj_count[lb] = 1
                    else:
                        obj_count[lb] += 1
                    pos = np.array(pos_lis[obj_id])*self.scale
                    item = QListWidgetItem(lb+f"_{obj_count[lb]}"+f"{tuple(pos.astype(int))}")
                    self.items.append(item)
                    self.ui.listWidget_rst.addItem(item)
                    lb_str = (lb if not visual_args["uniform_cls"] else "") + ((" "+str(round(prob_lis[obj_id],2))) if visual_args["draw_conf"] else "")
                    rect =  self.media_area.add_rec(lb_str,
                                                tuple(pos), 
                                                gt_labels[int(cls_lis[obj_id])], 
                                                prob_lis[obj_id], 
                                                self.labels, 
                                                obj_color[lb] if not visual_args["uniform_cls"] else (255, 0, 0),
                                                draw_label=visual_args["draw_label"],
                                                bold=visual_args["bold"],
                                                fond_sz=visual_args["fond_sz"],
                                                box_size=visual_args["box_sz"],
                                                item=item)
                    self.rects.append(rect)
        if self.ui.checkBox_sort.isChecked():
            self.ui.listWidget_rst.sortItems()
        obj_sum = sum(obj_count.values())
        for obj in obj_count:
            rst_txt += obj + ": " + str(obj_count[obj]) + "\n"
        self.ui.label_rst.setText(rst_txt)  # 显示结果
        self.ui.label_sum.setText(str(obj_sum))
        self.ui.checkBox_sort.setEnabled(True)
    
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
        detect_args["device"] = "0" if self.ui.radioButton_gpu.isChecked() else "cpu"
        return detect_args

    def get_visual_args(self):
        """
        获取可视化参数
        """
        visual_args = {}
        visual_args["draw_box"] = True if self.ui.checkBox_box.isChecked() else False
        visual_args["draw_conf"] = True if self.ui.checkBox_conf.isEnabled() and self.ui.checkBox_conf.isChecked() else False
        visual_args["bold"]  = True if self.ui.checkBox_bold.isChecked() else False
        visual_args["fond_sz"]  = int(float(self.ui.spinBox_font_size.text()))
        visual_args["box_sz"]  = int(float(self.ui.spinBox_box_sz.text()))
        visual_args["uniform_cls"] = True if self.ui.checkBox_uniform_cls.isEnabled() and self.ui.checkBox_uniform_cls.isChecked() else False
        visual_args["cls"] = []
        draw_label_1 = self.ui.checkBox_lb.isEnabled() and self.ui.checkBox_lb.isChecked()
        draw_label_2 = self.ui.checkBox_uniform_cls.isChecked() and self.ui.checkBox_conf.isChecked()
        visual_args["draw_label"] = True if draw_label_1 or draw_label_2 else False

        for i in range(self.cls_ui.ui.listWidget_tgt.count()):
            visual_args["cls"].append(self.cls_ui.ui.listWidget_tgt.item(i).text())
        return visual_args

    def draw_save(self, img:str, save_path:str):
        """
        绘制检测框
        """
        obj_color = {}
        temp_id = 0
        visual_args = self.get_visual_args()
        img = cv.imread(img)
        if visual_args["draw_box"]:
            for obj_id in range(len(self.pos_lis)):
                lb = gt_labels[self.cls_lis[obj_id]]
                if lb in visual_args["cls"]:
                    if lb not in obj_color:
                        obj_color[lb] = colors[temp_id] if temp_id < len(colors) else (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        temp_id += 1
                    pos = np.array(self.pos_lis[obj_id]) # xywh
                    lb_str = lb+((" "+str(round(self.prob_lis[obj_id],2))) if visual_args["draw_conf"] else "")
                    cv.rectangle(img, 
                                 (int(pos[0]), int(pos[1])), 
                                 (int(pos[0]+pos[2]), int(pos[1]+pos[3])), 
                                 (obj_color[lb][2], obj_color[lb][1], obj_color[lb][0]) if not visual_args["uniform_cls"] else (0, 0, 255), 
                                 visual_args.get("box_sz", 2))#该框宽不是像素个数，而是一个逻辑值，在不同分辨率下，其视觉大小不变，但这里当做像素值来用
                    if visual_args["draw_label"]:
                        cv.putText(img, 
                                   lb_str, 
                                   (int(pos[0]), int(pos[1]-5)), 
                                   cv.FONT_HERSHEY_SIMPLEX, 
                                   visual_args["fond_sz"]/15, # 改字体大小不是像素值，表示点数，其中1点（pt）等于1/72英寸，在这里除以15当像素值用
                                   (obj_color[lb][2], obj_color[lb][1], obj_color[lb][0]) if not visual_args["uniform_cls"] else (0, 0, 255), 
                                   1 if not visual_args["bold"] else 2, # 加粗的haul使用2，否则使用1
                                   cv.LINE_AA)
        cv.imwrite(save_path, img)

class ClassSelector(QDialog):
    def __init__(self, cls_lis:list[str], parent=None):
        super().__init__()
        self.ui = graphic_cls_ui.Ui_Dialog()
        self.ui.setupUi(self)
        self.init_ui(cls_lis)
        self.select_all = False
    
    def init_ui(self, cls_lis:list[str]):
        """
        初始化界面
        """
        # self.ui.listWidget_src.setSelectionRectVisible(True)
        self.ui.listWidget_src.addItems(cls_lis)
        self.ui.listWidget_tgt.addItems(cls_lis)
    
    def on_pushButton_locate_pressed(self):
        """
        选择指定类别
        """
        item = self.ui.lineEdit_locate.text().strip(" ")
        for i in range(self.ui.listWidget_src.count()):
            if self.ui.listWidget_src.item(i).text() == item:
                self.ui.listWidget_src.scrollToItem(self.ui.listWidget_src.item(i), QAbstractItemView.ScrollHint.EnsureVisible)
                self.ui.listWidget_src.setCurrentItem(self.ui.listWidget_src.item(i))
                break

    def on_pushButton_all_pressed(self):
        """
        全选
        """
        self.select_all = not self.select_all

        if self.select_all:
            self.ui.listWidget_src.selectAll()
        else:
            self.ui.listWidget_src.clearSelection()

    def on_pushButton_cls_ok_pressed(self):
        """
        确定选择
        """
        tgt_items = [self.ui.listWidget_tgt.item(i).text() for i in range(self.ui.listWidget_tgt.count())]
        for i in self.ui.listWidget_src.selectedItems():
            if i.text() not in tgt_items:
                self.ui.listWidget_tgt.addItem(i.text())
        self.ui.listWidget_src.clearSelection()
    
    def on_pushButton_cls_no_pressed(self):
        """
        取消选择
        """
        self.ui.listWidget_src.clearSelection()

    def on_pushButton_clr_pressed(self):
        """
        清空所有选择的标签
        """
        reply = QMessageBox.warning(self, "警告", "确定清空选择？", 
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                    QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.ui.listWidget_tgt.clear()
    
    def on_listWidget_src_itemDoubleClicked(self, item:QListWidgetItem):
        """ 
        双击项 
        """
        tgt_items = [self.ui.listWidget_tgt.item(i).text() for i in range(self.ui.listWidget_tgt.count())]
        if item.text() not in tgt_items:
            self.ui.listWidget_tgt.addItem(item.text())

    def on_listWidget_tgt_itemClicked(self, item):
        """
        取消单个选择的标签
        """
        row = self.ui.listWidget_tgt.row(item)
        self.ui.listWidget_tgt.takeItem(row) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectWindow()
    window.show()
    sys.exit(app.exec())