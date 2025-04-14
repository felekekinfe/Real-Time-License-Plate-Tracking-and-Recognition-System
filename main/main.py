from ultralytics import YOLO
import cv2
from sort import *
import numpy as np
from helper import get_car, read_license_plate
import pandas as pd
from threading import Thread
import queue

# Initialize tracker and models
vehicle_tracker = Sort()
coco_model = YOLO('model/yolov8n.pt')
license_plate_detector = YOLO('model/license_plate_detector.pt')

# Initialize queue for frames
frame_queue = queue.Queue(maxsize=3)  # Smaller queue for real-time

# Load video
cap = cv2.VideoCapture('output/pexels-taryn-elliott-5309381 (1080p).mp4')
vehicles = {2:'car',3:'motorcycle',5:'bus',7:'truck'}  # COCO classes: car, motorcycle, bus, truck
result = {}
frame_nmr = -1

# Producer thread: Read frames
def read_frames(cap):
    print("Starting read_frames")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                frame_queue.put((False, None))
                break
            frame = cv2.resize(frame, (640, 360))  # Lower res for speed
            frame_queue.put((True, frame))
    except Exception as e:
        print(f"Producer error: {e}")
        frame_queue.put((False, None))
    finally:
        cap.release()
        print("read_frames ended, video capture released")

# Start producer thread
producer = Thread(target=read_frames, args=(cap,), daemon=True)
producer.start()

# Consumer: Main thread processes and displays frames
try:
    while True:
        frame_nmr += 1
        try:
            ret, frame = frame_queue.get(timeout=0.1)  # Short timeout for responsiveness
        except queue.Empty:
            continue  # Skip if no frame yet, keep window responsive

        if not ret:
            print("Received end signal")
            break

        result[frame_nmr] = {}
        
        # Skip every 2nd frame to reduce load
        if frame_nmr % 2 != 0:
            continue

        # Detect vehicles
        detection = coco_model(frame, conf=0.6)[0]  # Higher threshold
        detections = []
        for det in detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if int(class_id) in vehicles:
                detections.append([x1, y1, x2, y2, score,class_id])

        # Track vehicles
        track_ids = vehicle_tracker.update(np.asarray(detections) if detections else np.empty((0, 6)))

        # Draw vehicle boxes and IDs
        for track in track_ids:
            x1, y1, x2, y2, track_id,class_id= map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, f"ID: {track_id} Type: {vehicles[class_id]}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect license plates every 5th frame
        license_plates = None
        #if frame_nmr % 2 == 0:
        license_plates = license_plate_detector(frame, conf=0.7)[0]

        if license_plates:
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = license_plate
                xc1, yc1, xc2, yc2, car_id = get_car(license_plate, track_ids)

                # Crop and process license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, 
                                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                                 cv2.THRESH_BINARY_INV, 11, 2)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                # Draw license plate box and text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red box
                if license_plate_text:
                    cv2.putText(frame, license_plate_text, (int(x1), int(y2+20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                
                    result[frame_nmr][car_id] = {
                        'car': {'bbox': [xc1, yc1, xc2, yc2]},
                        'license_plates': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

        # Display frame
        cv2.imshow('Vehicle and License Plate Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
            print("User quit")
            break

finally:
    # Clean up
    cv2.destroyAllWindows()
    while not frame_queue.empty():
        frame_queue.get_nowait()  # Clear queue
    producer.join(timeout=0.1)
    print("Cleanup complete")

# Process results and save to CSV
all_instances = {}
for frame, vehicles_dict in result.items():
    if not vehicles_dict:
        continue
    for car_id, info in vehicles_dict.items():
        lp_text = info['license_plates']['text']
        lp_score = info['license_plates']['bbox_score']
        key = (car_id, lp_text)
        lp_text_score = info['license_plates']['text_score']


        if lp_text_score > 0.6 and len(lp_text)>6:
            key = (car_id, lp_text)
            if key not in all_instances or lp_score > all_instances[key][2]:
                all_instances[key] = (frame, info, lp_score)


        

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

df = pd.DataFrame(filtered_data)
df.to_csv('output/plate.csv', index=False)
print("Results written to 'output/plate.csv'")