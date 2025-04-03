import tensorflow as tf
import cv2
import numpy as np

class YOLODetector:
    def __init__(self):
        self.model=tf.saved_model.load("http://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        self.classes={2:'car',7:'truck'}

    def detect(self,frame):
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_tensor=tf.convert_to_tensor([img],dtype=tf.uint8)

        detections=self.model(img_tensor)

        boxes=detections['detection_boxes'][0].numpy()
        scores=detections['detection_scores'][0].numpy()
        classes=detections['detection_classes'][0].numpy().astype(int)
        
        h,w=frame.shape[:2]
        vehicle_boxes=[]

        for i ,score in enumerate(scores):
            if score >0.5 and classes[i] in self.classes:
                y_min,x_min,y_max,x_max=boxes[i]
                box=[int(x_min*w),int(y_min*h),int(x_max*w),int(y_max*h)]

                vehicle_boxes.append((box,score,self.classes[classes[i]]))
        return vehicle_boxes

