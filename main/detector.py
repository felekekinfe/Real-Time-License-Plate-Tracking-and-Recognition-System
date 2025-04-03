from tensorflow import saved_model
import cv2
import numpy as np

class YOLODetector:
    def __init__(self):
        self.model=saved_model.load("http://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        self.classes={2:'car',7:'truck'}
        
