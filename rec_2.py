from PyQt6.QtWidgets import QGraphicsTextItem, QPushButton, QMainWindow, QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QLabel,  QGraphicsPixmapItem, QListWidgetItem 
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRectF, QEvent, QPointF
import sys
import numpy as np
from source.ui.rec_ui import Ui_MainWindow

view_click = 0
rect_click = 0
rec_record = None

class RecWidget(QGraphicsRectItem):
    def __init__(self, pos:tuple, obj_cls:str, prob:float, labels:list[QLabel], color:tuple, size:int=2, item:QListWidgetItem=None, parent=None):
        super().__init__(parent)
        self.obj_pos = pos
        self.obj_cls = obj_cls
        self.prob = prob
        self.labels = labels
        self.color = color
        self.size = size
        self.setRect(QRectF(*pos))
        self.setPen(QPen(QColor(*color), size))
        self.press = False
        self.setAcceptHoverEvents(True)
        self.item = item # 用于记录对应的QListWidgetItem

    def hoverEnterEvent(self, event):  
        super().hoverEnterEvent(event)   
        self.setPen(QPen(Qt.GlobalColor.red, self.size+2))
    
    def hoverLeaveEvent(self, event):  
        super().hoverLeaveEvent(event)  
        if not self.press:  
            self.setPen(QPen(QColor(*self.color), self.size))
    
    def mousePressEvent(self, event):  
        global view_click, rect_click, rec_record
        super().mousePressEvent(event) 
        rect_click += 1
        print("after rec_click")
        print("view_click:", view_click)
        print("rect_click:", rect_click)
        if rec_record:
            rec_record.setPen(QPen(QColor(*self.color), self.size))
            rec_record.press = False
        rec_record = self
        self.press = True  
        self.setPen(QPen(Qt.GlobalColor.red, self.size+2))
        self.info_display = self.InfoDisplay()
        self.item.setSelected(True)

    def InfoDisplay(self):
        self.labels[0].setText(self.obj_cls)
        self.labels[1].setText(str(round(self.prob, 2)))
        self.labels[2].setText(str(int(self.obj_pos[0])))
        self.labels[3].setText(str(int(self.obj_pos[1])))
        self.labels[4].setText(str(int(self.obj_pos[2])))
        self.labels[5].setText(str(int(self.obj_pos[3])))

class Graficview(QGraphicsView):
    global view_click

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.resize(600, 600)
        self.rect_items = []
        self.text_items = []
        self.pixmap_item = None
        self.initUI()

    def initUI(self):  
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)       
    
    def add_img(self, img_p, scale_w, scale_h):  
        """
        添加图片
        :param img_p: 图片路径
        :param scale_w: 图片缩放宽度
        :param scale_h: 图片缩放高度
        """
        # 添加图像前把前面的图像以及框清除
        self.scene_.clear() 
        if self.pixmap_item:
            # self.scene_.removeItem(self.pixmap_item)
            del self.pixmap_item 
        self.remove_rec()

        pixmap = QPixmap(img_p).scaled(scale_w, scale_h)  # 替换为你的图片路径  , Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        self.pixmap_item = QGraphicsPixmapItem(pixmap)  
        self.scene_.addItem(self.pixmap_item)
    
    def add_rec(self,
                lb_str:str, 
                pos:tuple, 
                obj_cls:str, 
                prob:float, 
                labels:list[QLabel], 
                color:tuple, 
                draw_label=True, 
                bold=True,
                fond_sz=8,
                box_size=2, 
                item=None)->RecWidget:
        """
        添加矩形框
        :param pos: 矩形框的位置和大小
        :param obj_cls: 物体类别
        :param prob: 物体概率
        :param labels: 显示信息的标签
        """
        # print(self.scene_.items())
        # 设置矩形框的位置和大小 
        if draw_label: 
            text_item = QGraphicsTextItem(lb_str)  
            # 设置文本的属性（可选）  
            font = QFont()  
            font.setPointSize(fond_sz) 
            if bold:
                font.setWeight(QFont.Weight.Bold)
            text_item.setFont(font)  
            text_item.setDefaultTextColor(QColor(*color))
            
            # 设置文本项在场景中的位置（例如，设置到 (100, 50)）  
            text_item.setPos(QPointF(pos[0], pos[1]-20))  
            self.text_items.append(text_item) 
            self.scene_.addItem(text_item)  
        rect_item = RecWidget(pos, obj_cls, prob, labels, color, box_size, item)  
        self.rect_items.append(rect_item)  
        self.scene_.addItem(rect_item)  
        return rect_item
    
    def remove_rec(self):  
        """
        移除矩形框
        """
        for text_item in self.text_items:
            if text_item in self.scene_.items():
                self.scene_.removeItem(text_item)
            del text_item
        for rect_item in self.rect_items:
            if rect_item in self.scene_.items():
                self.scene_.removeItem(rect_item)
            del rect_item
    
    def mousePressEvent(self, event):  
        global view_click, rect_click, rec_record
        super().mousePressEvent(event)
        view_click += 1
        print("after view_click")
        print("view_click:", view_click)
        print("rect_click:", rect_click)
        if view_click != rect_click:
            if rec_record:
                rec_record.setPen(QPen(QColor(*rec_record.color), rec_record.size))
                rec_record.press = False
                rec_record = None
        view_click = 0
        rect_click = 0
        

class btn(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def setHoveredLabel(self, label):  
        self.hovered_label = label  

  
    def enterEvent(self, event: QEvent) -> None:  
        super().enterEvent(event)  # 调用基类的enterEvent  
        if self.hovered_label:  
            self.hovered_label.setText('1')  # 更新标签的文本 
    
    def leaveEvent(self, event: QEvent) -> None:  
        super().leaveEvent(event)  # 调用基类的leaveEvent  
        if self.hovered_label:  
            self.hovered_label.setText('0')  # 更新标签的文本 

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()
    
    def initUI(self):  
        self.bt = btn(self) 

        self.bt.setHoveredLabel(self.ui.label) 
        self.bt.setGeometry(660,210, 75, 24)  

        self.view = Graficview(r"source\image\street.jpg", self)
        self.view.add_rec((100, 100, 200, 200), self.ui.label)
        self.view.add_rec((100, 350, 200, 100), self.ui.label)

if __name__ == '__main__':  
    app = QApplication([])  
    view = MainWindow()
    view.show()  
    # view.show()  
    sys.exit(app.exec())