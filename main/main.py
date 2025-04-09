from ultralytics import YOLO
import cv2
from sort import *
import numpy as np
from helper import get_car, read_license_plate
import pandas as pd
import time

vehicle_tracker = Sort()
# Load models
coco_model = YOLO('model/yolov8n.pt')
license_plate_detector = YOLO('model/license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('ggg.jpeg')
result = {}
ret = True
frame_nmr = -1
vehicles = [2, 3, 5, 7]  # COCO classes: car (2), motorcycle (3), bus (5), truck (7)

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if frame_nmr<10:  # Process all frames
        result[frame_nmr] = {}
        # Detect vehicles
        detection = coco_model(frame)[0]
        detection.show()  # Uncomment to visualize
        print(f'detection {detection}')

        detections = []
        for detection in detection.boxes.data.tolist():
            print(f'detection loop {detection}')
            x1, y1, x2, y2, score, class_id = detection
            print(f'class id; {class_id}')
            if int(class_id) in vehicles:
                detections.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = vehicle_tracker.update(np.asarray(detections))
        print(f'track_ids {track_ids}')

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        license_plates.show()  # Uncomment to visualize

        for license_plate in license_plates.boxes.data.tolist():
            print(f'license_plate {license_plate}')
            x1, y1, x2, y2, score, _ = license_plate

            # Assign plate to vehicle
            xc1, yc1, xc2, yc2, car_id = get_car(license_plate, track_ids)

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            
            # Process the plate image
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate text
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                result[frame_nmr][car_id] = {
                    'car': {'bbox': [xc1, yc1, xc2, yc2]},
                    'license_plates': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }
        
      

print(f'RESULT: {result}')

# Group by car_id and lp_text, keeping the highest lp_bbox_score for duplicates
all_instances = {}  # (car_id, lp_text) -> (frame, info, lp_bbox_score)
for frame, vehicles_dict in result.items():
    if not vehicles_dict:
        continue
    for car_id, info in vehicles_dict.items():
        lp_text = info['license_plates']['text']
        lp_score = info['license_plates']['bbox_score']
        key = (car_id, lp_text)
        if key not in all_instances or lp_score > all_instances[key][2]:
            all_instances[key] = (frame, info, lp_score)

# Prepare filtered data
filtered_data = []
for (car_id, lp_text), (frame, info, _) in all_instances.items():
    car_bbox = info['car']['bbox']
    lp_info = info['license_plates']
    filtered_data.append({
        'frame': int(frame),
        'car_id': float(car_id),
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

# Write to CSV
df = pd.DataFrame(filtered_data)
df.to_csv('output/plate.csv', index=False)
print("All unique license plates per car_id with highest lp_bbox_score written to 'output/plate_all_plates.csv'")

cap.release()