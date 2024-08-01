import numpy as np
import torch
import cv2 as cv
import time 
from ultralytics import YOLO


def detect(img_path:str, model:YOLO, detect_args): #,labels:bool=True, boxes:bool=True, conf:bool=True
    """
    检测图像中的物体
    :param img: 输入图像
    :return: 检测到的物体
    改进：多张图片，多进程,画框与显示解耦
    """
    img = cv.imread(img_path)
    results = model.predict(img, save=False, **detect_args)
    pos = results[0].boxes.xywh[:, :2] - results[0].boxes.xywh[:, 2:]/2
    pos = torch.cat([pos, results[0].boxes.xywh[:, 2:]], dim=1)
    return results[0].boxes.cls.cpu().numpy().tolist(), results[0].boxes.conf.cpu().numpy().tolist(), pos.cpu().numpy().tolist(), results[0].speed["inference"]

if __name__ == '__main__':
    img_p = r"source\image\bus.jpg"
    img = cv.imread(img_p)
    detect(img)
  