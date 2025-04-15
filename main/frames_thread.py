import queue
import cv2



# Initialize queue for frames
frame_queue = queue.Queue(maxsize=3)  # Smaller queue for real-time
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