from ultralytics import YOLO
import cv2
from sort.sort import *
import numpy as np
from helper import get_car,read_license_plate


vehicle_tracker=Sort()
#load model
coco_model=YOLO('yolov8n.pt')
license_plate_detector=YOLO('license_plate_detector.pt')

#load video
cap=cv2.VideoCapture('output/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4')
result={}
ret=True
frame_nmr=-1
vehicles=[2,3,5,7]
while ret:
    frame_nmr+=1

    ret,frame=cap.read()
    if ret and frame_nmr<10:
        result[frame_nmr]={}
        #detect vehicles
        detection=coco_model(frame)[0]
        detections=[]

        for detection in detection.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=detection
            if int(class_id) in vehicles:
                detections.append([x1,y1,x2,y2,score])

        #track vehicles
        track_ids=vehicle_tracker.update(np.asarray(detections))

        #detecte license plate
        license_paltes=license_plate_detector(frame)[0]

        for license_plate in license_paltes.boxes.data.tolist():
            x1,y1,x2,y2,score,car_id=license_plate

            #assign plate to vehicle
            xc1,yc1,xc2,yc2=get_car(license_plate,track_ids)

            #crop
            license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2), :]
            
            #process the plate image
            license_plate_crop_gray=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
            _,license_plate_crop_thresh=cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV)

            #read license_plate
            license_plate_text,license_plate_text_score=read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                result[frame_nmr][car_id]={'car':{'bbox':[xc1,yc1,xc2,yc2]},
                                           'license_plates':{'bbox':[x1,y1,x2,y2],
                                                             'text':license_plate_text,
                                                             'bbox_score':score,
                                                             'text_score':license_plate_text_score}

                }
            #write result

    
