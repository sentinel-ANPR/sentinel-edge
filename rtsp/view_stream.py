import cv2
import sys
import time

# RTSP URL
RTSP_URL = "rtsp://127.0.0.1:8554/stream"

def view_stream():
    """Connect to and display RTSP stream."""
    print(f"Connecting to {RTSP_URL}...")
    
    # Open video capture with RTSP
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream")
        print("Make sure the streaming server is running")
        sys.exit(1)
    
    print("Connected successfully!")
    print("Press 'q' to quit, 'r' to reconnect")
    
    reconnect = False
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Warning: Could not read frame, attempting reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        # Display frame
        cv2.imshow('RTSP Stream Viewer', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('r'):
            print("Reconnecting...")
            cap.release()
            cv2.destroyAllWindows()
            time.sleep(0.5)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        view_stream()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
