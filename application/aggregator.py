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

from telemetry import (
    record_job_completed,
    record_job_timeout,
    record_reclaim,
    record_upload_latency,
)

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
        self.job_timeout_sec = JOB_TIMEOUT_SEC

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

    def fill_missing_results(self, vehicle_type, results):
        defaults = {
            "ocr": "N/A",
            "color": "unknown|#000000",
            "logo": "Unknown",
            "violation": "0",
        }
        expected = get_expected_workers(vehicle_type)
        merged = dict(results)
        for worker in expected:
            if worker not in merged:
                merged[worker] = defaults.get(worker, "N/A")
        return merged
    
    def cleanup_files(self, frame_path, plate_path, logo_path=None):
        to_delete = [frame_path, plate_path, logo_path]
        
        for path in to_delete:
            # Check for both Python None and string "None"/"N/A"
            if path and str(path) not in ["None", "N/A", ""] and os.path.exists(path):
                try:
                    os.remove(path)
                    # self.log_agg(f"Deleted: {os.path.basename(path)}")
                except Exception as e:
                    self.log_agg(f"Cleanup Error on {path}: {e}")

    def report_to_central(self, job_data, frame_path, plate_path, logo_path=None):
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
        upload_start = time.time()
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

            # open keyframe
            files["keyframe_file"] = open(frame_path, "rb")
            
            # open plate 
            if plate_path and os.path.exists(plate_path):
                files["plate_file"] = open(plate_path, "rb")
            
            # open logo
            if logo_path and logo_path != "N/A" and os.path.exists(logo_path):
                files["logo_file"] = open(logo_path, "rb")

            self.log_agg(f"Uploading {job_data['vehicle_id']} to Central...")

            response = requests.post(endpoint, data=payload, files=files, timeout=10)
            
            success = response.status_code == 200
            record_upload_latency(self.location, time.time() - upload_start, success)

            if success:
                self.log_agg(f"Success: {job_data['vehicle_id']}")
                # remove files from disk
                self.cleanup_files(frame_path, plate_path, logo_path)
                return True
            return False

        except Exception as e:
            record_upload_latency(self.location, time.time() - upload_start, False)
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
                reclaimed = reclaim_pending_messages(
                    self.r, AGGREGATOR_GROUP, "edge_agg_1", VEHICLE_RESULTS_STREAM, ACK_TIMEOUT
                )
                if reclaimed:
                    record_reclaim(VEHICLE_RESULTS_STREAM, AGGREGATOR_GROUP, len(reclaimed))
                    messages = [(VEHICLE_RESULTS_STREAM, reclaimed)]
                else:
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

                        # take frame_path, plate_path, and logo_path from the redis message.
                        if job_id not in self.pending_jobs:
                            self.pending_jobs[job_id] = {
                                "results": {},
                                "result_ids": set(),
                                "vehicle_id": f.get("vehicle_id"),
                                "frame_path": f.get("frame_path") or f.get("keyframe_path"), # check both keys
                                "plate_path": f.get("plate_path"),
                                "logo_path": None,
                                "timestamp": f.get("timestamp") or datetime.datetime.now().isoformat(),
                                "created_ts": time.time(),
                                "job_msg_id": f.get("job_msg_id")
                            }
                        elif f.get("job_msg_id"):
                            self.pending_jobs[job_id]["job_msg_id"] = f.get("job_msg_id")
                        
                        # update paths if they appear in later worker messages
                        current_path = f.get("frame_path") or f.get("keyframe_path")
                        if current_path: self.pending_jobs[job_id]["frame_path"] = current_path
                        
                        current_plate = f.get("plate_path")
                        if current_plate: self.pending_jobs[job_id]["plate_path"] = current_plate
                        
                        # update logo_path when logo worker responds
                        current_logo = f.get("logo_path")
                        if current_logo and current_logo != "N/A":
                            self.pending_jobs[job_id]["logo_path"] = current_logo

                        self.pending_jobs[job_id]["results"][worker] = result
                        self.pending_jobs[job_id]["result_ids"].add(msg_id)
                        v_type = job_id.split("_")[0]
                        expected = get_expected_workers(v_type)

                        ready = set(self.pending_jobs[job_id]["results"].keys()) >= set(expected)
                        timed_out = (time.time() - self.pending_jobs[job_id]["created_ts"]) >= self.job_timeout_sec

                        if ready or timed_out:
                            if timed_out and not ready:
                                self.log_agg(f"Job timeout for {job_id}; using fallback results")
                                record_job_timeout(self.location, v_type)

                            res = self.fill_missing_results(v_type, self.pending_jobs[job_id]["results"])
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
                            if self.report_to_central(
                                job_data, 
                                self.pending_jobs[job_id]["frame_path"], 
                                self.pending_jobs[job_id]["plate_path"],
                                self.pending_jobs[job_id].get("logo_path")
                            ):
                                record_job_completed(
                                    self.location,
                                    v_type,
                                    time.time() - self.pending_jobs[job_id]["created_ts"]
                                )
                                result_ids = list(self.pending_jobs[job_id]["result_ids"])
                                if result_ids:
                                    self.r.xack(VEHICLE_RESULTS_STREAM, AGGREGATOR_GROUP, *result_ids)
                                    for _id in result_ids:
                                        self.r.xdel(VEHICLE_RESULTS_STREAM, _id)

                                job_msg_id = self.pending_jobs[job_id].get("job_msg_id")
                                if job_msg_id:
                                    self.r.xdel(VEHICLE_JOBS_STREAM, job_msg_id)
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