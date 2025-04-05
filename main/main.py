from ultralytics import YOLO
import cv2
from sort import *
import numpy as np
from helper import get_car,read_license_plate,write_csv
import pandas as pd

vehicle_tracker=Sort()
#load model
coco_model=YOLO('yolov8n.pt')
license_plate_detector=YOLO('license_plate_detector.pt')

#load video
cap=cv2.VideoCapture('output/Traffic Control CCTV.mp4')
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
        #detection.show()
        print(f'detection {detection}')

        detections=[]

        for detection in detection.boxes.data.tolist():
            print(f'detection loop {detection}')
            x1,y1,x2,y2,score,class_id=detection
            print(f'class id; {class_id}')
            if int(class_id) in vehicles:
                detections.append([x1,y1,x2,y2,score])

        #track vehicles
        track_ids=vehicle_tracker.update(np.asarray(detections))
        print(f'track_ids {track_ids}')

        #detecte license plate
        license_plates=license_plate_detector(frame)[0]
        #license_plates.show()

        for license_plate in license_plates.boxes.data.tolist():
            print(f'license_plate {license_plate}')
            x1,y1,x2,y2,score,car_id=license_plate

            #assign plate to vehicle
            xc1,yc1,xc2,yc2,car_id=get_car(license_plate,track_ids)

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
print(f'RESULT: {result}')
data = []
for frame, vehicles_dict in result.items():
    for car_id, info in vehicles_dict.items():
        car_bbox = info['car']['bbox']
        lp_info = info['license_plates']
        data.append({
            'frame': int(frame),  # Convert to int for cleaner CSV
            'car_id': float(car_id),  # Keep as float to match your keys
            'car_x1': car_bbox[0],
            'car_y1': car_bbox[1],
            'car_x2': car_bbox[2],
            'car_y2': car_bbox[3],
            'lp_x1': lp_info['bbox'][0],
            'lp_y1': lp_info['bbox'][1],
            'lp_x2': lp_info['bbox'][2],
            'lp_y2': lp_info['bbox'][3],
            'lp_text': lp_info['text'],
            'lp_bbox_score': lp_info['bbox_score'],
            'lp_text_score': lp_info['text_score']
        })

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('output/plate.csv', index=False)
#write result
#write_csv(result,'output/plate.csv')
    
