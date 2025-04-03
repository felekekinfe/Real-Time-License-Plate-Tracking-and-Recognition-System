import cv2
import time

def get_video(video_path):
    cap=cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        raise ValueError('error on opening video source')
    return cap


def write_log(file_path,message):
    with open(file_path,'a') as f:
        f.write(f'{time.strftime('%H:%M:%S')}-{message}\n')

def save_frame(frame,output_path='output/demo.mp4',writer=None):
    if writer is None:
        h,w=frame.shape[:2]
        writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'))
    writer.write(frame)
    return writer


