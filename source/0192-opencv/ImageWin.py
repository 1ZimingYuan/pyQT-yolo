# Form implementation generated from reading ui file 'ImageWin.ui'
#
# Created by: PyQt6 UI code generator 6.2.3
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ImageForm(object):
    def setupUi(self, ImageForm):
        ImageForm.setObjectName("ImageForm")
        ImageForm.resize(842, 555)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ImageForm.sizePolicy().hasHeightForWidth())
        ImageForm.setSizePolicy(sizePolicy)
        ImageForm.setMaximumSize(QtCore.QSize(842, 16777215))
        self.formLayout = QtWidgets.QFormLayout(ImageForm)
        self.formLayout.setObjectName("formLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_OpenImage = QtWidgets.QPushButton(ImageForm)
        self.pushButton_OpenImage.setObjectName("pushButton_OpenImage")
        self.verticalLayout.addWidget(self.pushButton_OpenImage)
        self.label_originImage = QtWidgets.QLabel(ImageForm)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_originImage.sizePolicy().hasHeightForWidth())
        self.label_originImage.setSizePolicy(sizePolicy)
        self.label_originImage.setMaximumSize(QtCore.QSize(400, 16777215))
        self.label_originImage.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_originImage.setObjectName("label_originImage")
        self.verticalLayout.addWidget(self.label_originImage)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_BlurImage = QtWidgets.QPushButton(ImageForm)
        self.pushButton_BlurImage.setObjectName("pushButton_BlurImage")
        self.verticalLayout_2.addWidget(self.pushButton_BlurImage)
        self.label_blueImage = QtWidgets.QLabel(ImageForm)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_blueImage.sizePolicy().hasHeightForWidth())
        self.label_blueImage.setSizePolicy(sizePolicy)
        self.label_blueImage.setMaximumSize(QtCore.QSize(400, 16777215))
        self.label_blueImage.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_blueImage.setObjectName("label_blueImage")
        self.verticalLayout_2.addWidget(self.label_blueImage)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.ItemRole.SpanningRole, self.horizontalLayout)
        self.blurSlider = QtWidgets.QSlider(ImageForm)
        self.blurSlider.setEnabled(False)
        self.blurSlider.setMinimum(1)
        self.blurSlider.setMaximum(100)
        self.blurSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.blurSlider.setObjectName("blurSlider")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.SpanningRole, self.blurSlider)

        self.retranslateUi(ImageForm)
        QtCore.QMetaObject.connectSlotsByName(ImageForm)

    def retranslateUi(self, ImageForm):
        _translate = QtCore.QCoreApplication.translate
        ImageForm.setWindowTitle(_translate("ImageForm", "Form"))
        self.pushButton_OpenImage.setText(_translate("ImageForm", "打开图片"))
        self.label_originImage.setText(_translate("ImageForm", "请打开一个图片，图片目录需要纯英文或者字母"))
        self.pushButton_BlurImage.setText(_translate("ImageForm", "模糊图像"))
        self.label_blueImage.setText(_translate("ImageForm", "点击模糊图像"))
