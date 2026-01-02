import os
import sys
import time
import json
import datetime
import requests
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from db_redis.sentinel_redis_config import *

load_dotenv()

# set timezone to IST
os.environ["TZ"] = "Asia/Kolkata"
if sys.platform != "win32":
    time.tzset()

class ResultAggregator:
    # bundles worker results and uploads images to Central Server
    
    def __init__(self):
        self.pending_jobs = defaultdict(dict)
        self.r = get_redis_connection()
        self.central_url = os.getenv("CENTRAL_API_URL")
        self.location = os.getenv("LOCATION", "UNKNOWN")

    def log_agg(self, message):
        YELLOW = "\033[93m"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{YELLOW}[ Aggregator] {timestamp} | {message}")

    def parse_color_result(self, result):
        if isinstance(result, bytes): result = result.decode('utf-8')
        if '|' in result:
            color_name, hex_code = result.split('|', 1)
            return color_name.strip(), hex_code.strip()
        return result.strip(), "#000000"
    
    def cleanup_files(self, frame_path, plate_path):
        try:
            if frame_path and os.path.exists(frame_path):
                os.remove(frame_path)
            
            if plate_path and os.path.exists(plate_path):
                os.remove(plate_path)
        except Exception as e:
            self.log_agg(f"Cleanup Error: {e}")

    def report_to_central(self, job_data, frame_path, plate_path):
        # upload physical binary files to the central server
        if not self.central_url:
            self.log_agg("CENTRAL_API_URL not configured")
            return False

        endpoint = f"{self.central_url}/api/ingest/vehicle-complete"
        
        # validation: frame_path is mandatory for upload 
        if not frame_path or frame_path in ["None", "", b"None"]:
            self.log_agg(f"No valid frame path for {job_data['vehicle_id']}")
            return False

        files = {}
        try:
            # metadata payload
            payload = {
                "vehicle_id": job_data["vehicle_id"],
                "vehicle_type": job_data["vehicle_type"],
                "vehicle_number": job_data["vehicle_number"],
                "color": f"{job_data['color']}|{job_data['color_hex']}",
                "model": job_data["model"],
                "violation_type": job_data["violation_type"],
                "location": self.location,
                "timestamp": job_data["timestamp"]
            }

            # repalce both with redis
            # open keyframe
            files["keyframe_file"] = open(frame_path, "rb")
            
            # open plate 
            if plate_path and os.path.exists(plate_path):
                files["plate_file"] = open(plate_path, "rb")

            self.log_agg(f"Uploading {job_data['vehicle_id']} to Central...")
            response = requests.post(endpoint, data=payload, files=files, timeout=10)
            
            if response.status_code == 200:
                self.log_agg(f"Success: {job_data['vehicle_id']}")
                # remove file from disk
                self.cleanup_files(frame_path, plate_path)
                return True
            return False

        except Exception as e:
            self.log_agg(f"Upload failed: {e}")
            return False
        finally:
            for f in files.values(): f.close()

    def process_results(self):
        self.log_agg(f"Edge Aggregator started for {self.location}")
        try:
            self.r.xgroup_create(VEHICLE_RESULTS_STREAM, AGGREGATOR_GROUP, id="0", mkstream=True)
        except: pass

        while True:
            try:
                messages = self.r.xreadgroup(AGGREGATOR_GROUP, "edge_agg_1",
                    {VEHICLE_RESULTS_STREAM: ">"}, count=10, block=1000)

                if not messages: continue

                for _, msgs in messages:
                    for msg_id, fields in msgs:
                        f = {k.decode('utf-8') if isinstance(k, bytes) else k: 
                             v.decode('utf-8') if isinstance(v, bytes) else v 
                             for k, v in fields.items()}

                        job_id = f.get("job_id")
                        worker = f.get("worker")
                        result = f.get("result")
                        
                        if not job_id: continue

                        # take frame_path and plate_path from the redis message.
                        if job_id not in self.pending_jobs:
                            self.pending_jobs[job_id] = {
                                "results": {},
                                "vehicle_id": f.get("vehicle_id"),
                                "frame_path": f.get("frame_path") or f.get("keyframe_path"), # check both keys
                                "plate_path": f.get("plate_path"),
                                "timestamp": f.get("timestamp") or datetime.datetime.now().isoformat()
                            }
                        
                        # i[pdate paths if they appear in later worker messages
                        current_path = f.get("frame_path") or f.get("keyframe_path")
                        if current_path: self.pending_jobs[job_id]["frame_path"] = current_path
                        
                        current_plate = f.get("plate_path")
                        if current_plate: self.pending_jobs[job_id]["plate_path"] = current_plate

                        self.pending_jobs[job_id]["results"][worker] = result
                        v_type = job_id.split("_")[0]
                        expected = get_expected_workers(v_type)

                        if set(self.pending_jobs[job_id]["results"].keys()) >= set(expected):
                            res = self.pending_jobs[job_id]["results"]
                            c_name, c_hex = self.parse_color_result(res.get("color", "unknown|#000000"))
                            
                            job_data = {
                                "vehicle_id": self.pending_jobs[job_id]["vehicle_id"],
                                "vehicle_type": v_type,
                                "vehicle_number": res.get("ocr", "N/A"),
                                "color": c_name,
                                "color_hex": c_hex,
                                "model": res.get("logo", "Unknown"),
                                "violation_type": int(res.get("violation", 0)),
                                "timestamp": self.pending_jobs[job_id]["timestamp"]
                            }

                            # upload using the absolute paths stored in memory
                            if self.report_to_central(job_data, self.pending_jobs[job_id]["frame_path"], self.pending_jobs[job_id]["plate_path"]):
                                self.r.xack(VEHICLE_RESULTS_STREAM, AGGREGATOR_GROUP, msg_id)
                                del self.pending_jobs[job_id]

            except Exception as e:
                self.log_agg(f"Error: {e}")
                time.sleep(1)

def start_edge_aggregator():
    # entry point to initialize and start the ResultAggregator loop.
    print(f"\n--- Starting Sentinel Edge Aggregator ---")
    print(f"Node ID: {os.getenv('NODE_ID')}")
    print(f"Location: {os.getenv('LOCATION')}")
    print(f"Central Server: {os.getenv('CENTRAL_API_URL')}")
    
    # initialize the Aggregator Engine
    aggregator = ResultAggregator()

    try:
        # start the blocking process loop
        # this polls local Redis and POSTS results + images to Central
        aggregator.process_results()
        
    except KeyboardInterrupt:
        print("\nEdge Aggregator shutting down...")
    except Exception as e:
        print(f"Critical error in Edge Aggregator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_edge_aggregator()