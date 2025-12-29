import subprocess
import time
import cv2
import sys
import os
import signal
import argparse


# Paths
MEDIA_FILE = "in1.mp4"
MEDIAMTX_BINARY = "./mediamtx"  # path to MediaMTX binary
RTSP_URL = "rtsp://127.0.0.1:8554/stream"



def start_mediamtx():
    """Start MediaMTX server."""
    return subprocess.Popen([MEDIAMTX_BINARY, "mediamtx.yml"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)



def start_stream():
    """Start streaming MP4 to RTSP using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", MEDIA_FILE,
        "-c", "copy",                   
        "-rtsp_transport", "tcp",
        "-f", "rtsp",
        RTSP_URL
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)



def view_stream():
    """View RTSP stream using OpenCV."""
    print("Connecting to RTSP stream...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream")
        return
    
    print("Stream viewer opened. Press 'q' to close window.")
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        cv2.imshow('RTSP Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTSP streaming server')
    parser.add_argument('--view', action='store_true', help='Open CV2 viewer window to display stream')
    args = parser.parse_args()
    
    mediamtx_proc = None
    ffmpeg_proc = None
    
    try:
        # Start MediaMTX
        print("Starting MediaMTX server...")
        mediamtx_proc = start_mediamtx()
        time.sleep(1)


        # Start streaming MP4
        print("Starting MP4 stream...")
        ffmpeg_proc = start_stream()
        time.sleep(2)


        # Open viewer if flag is set
        if args.view:
            view_stream()
        else:
            # Keep running
            print("Stream active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)


    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup processes
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        if mediamtx_proc:
            mediamtx_proc.terminate()
