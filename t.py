import sys  
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QScrollArea, QVBoxLayout, QWidget  
from PyQt6.QtCore import Qt  
  
class ScrollableLabelDemo(QMainWindow):  
    def __init__(self):  
        super().__init__()  
  
        # 设置主窗口的标题和大小  
        self.setWindowTitle('Scrollable Label Demo')  
        self.setGeometry(100, 100, 100, 100)  
  
        # 创建一个QWidget作为中央小部件  
        central_widget = QWidget(self)  
        self.setCentralWidget(central_widget)  
  
        # 使用QVBoxLayout管理布局  
        layout = QVBoxLayout(central_widget)  
  
        # 创建一个QLabel并设置长文本  
        long_text_label = QLabel("这里是一个非常长的文本，以至于它的高发士大夫鬼地方公司的人瑟夫撒eraser发售日瓦尔瓦而微软围绕围绕微软不能一次性在QLabel中完全显示。我们需要使用QScrollArea来让QLabel的内容支持滚动查看。")  
        long_text_label.setWordWrap(True)  # 启用自动换行  
  
        # 创建一个QScrollArea  
        scroll_area = QScrollArea(central_widget)  
        scroll_area.setWidgetResizable(True)  # 允许内部小部件调整大小  
  
        # 将QLabel设置为QScrollArea的widget  
        scroll_area.setWidget(long_text_label)  
  
        # 将QScrollArea添加到布局中  
        layout.addWidget(scroll_area)  
  
if __name__ == '__main__':  
    app = QApplication(sys.argv)  
    demo = ScrollableLabelDemo()  
    demo.show()  
    sys.exit(app.exec())