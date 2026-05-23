import os
# Set timezone to IST
os.environ["TZ"] = "Asia/Kolkata"
import time
time.tzset()
import threading
import subprocess
import signal
import sys
import psutil
import requests
import gc
from aggregator import ResultAggregator
from db_redis.sentinel_redis_config import *
from dotenv import load_dotenv
load_dotenv()

from telemetry import start_metrics_pusher, update_redis_metrics


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# --- Configuration for Auto-Scaling ---
SCALING_CONFIG = {
    "ocr": {
        "min": _env_int("OCR_MIN_WORKERS", 1),
        "max": _env_int("OCR_MAX_WORKERS", 2),
        "threshold": _env_int("OCR_SCALE_THRESHOLD", 20),
        "script": "ocr/ocr_worker_paddle.py",
        "color": "92",
        "group": OCR_GROUP,
    },
    "color": {
        "min": _env_int("COLOR_MIN_WORKERS", 1),
        "max": _env_int("COLOR_MAX_WORKERS", 2),
        "threshold": _env_int("COLOR_SCALE_THRESHOLD", 20),
        "script": "color_detection/color_worker_yolo.py",
        "color": "94",
        "group": COLOR_GROUP,
    },
    "logo": {
        "min": _env_int("LOGO_MIN_WORKERS", 1),
        "max": _env_int("LOGO_MAX_WORKERS", 2),
        "threshold": _env_int("LOGO_SCALE_THRESHOLD", 20),
        "script": "logo_detection/logo_worker.py",
        "color": "95",
        "group": LOGO_GROUP,
    },
    "violation": {
        "min": _env_int("VIOLATION_MIN_WORKERS", 1),
        "max": _env_int("VIOLATION_MAX_WORKERS", 2),
        "threshold": _env_int("VIOLATION_SCALE_THRESHOLD", 15),
        "script": "violation_detection/violation_worker.py",
        "color": "97",
        "group": VIOLATION_GROUP,
    },
}

MAX_CPU_PERCENT = _env_float("MAX_CPU_PERCENT", 85.0)
MAX_RAM_PERCENT = _env_float("MAX_RAM_PERCENT", 85.0)
RESOURCE_LOG_INTERVAL = _env_int("RESOURCE_LOG_INTERVAL", 15)

class SentinelOrchestrator:
    def __init__(self):
        self.processes = {} 
        self.worker_groups = {k: [] for k in SCALING_CONFIG.keys()} 
        self.ingress_processes = []
        self.last_resource_log_ts = 0.0
        
        self.r = get_redis_connection()
        self.shutdown_requested = False
        self.shutdown_lock = threading.Lock()
        
        self.node_id = os.getenv("NODE_ID", "UNNAMED_NODE")
        self.central_url = os.getenv("CENTRAL_API_URL")
        self.location = os.getenv("LOCATION", "DEFAULT_LOCATION")
        
        self.rtsp_streams = os.getenv("RTSP_STREAMS", "").split(",")
        if not self.rtsp_streams or self.rtsp_streams == ['']:
            single_stream = os.getenv("RTSP_STREAM")
            if single_stream:
                self.rtsp_streams = [single_stream]
            else:
                print("\nERROR: No RTSP_STREAMS or RTSP_STREAM found in .env")
                sys.exit(1)

        print(f"Orchestrator initialized for location: {self.location}")
        print(f"Detected {len(self.rtsp_streams)} video stream(s).")

    def start_heartbeat(self):
        def heartbeat_loop():
            while not self.shutdown_requested:
                try:
                    requests.post(
                        f"{self.central_url}/api/monitor/heartbeat", 
                        json={"node_id": self.node_id}, 
                        timeout=5
                    )
                except Exception:
                    pass
                time.sleep(10) 
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        print(f"[{self.node_id}] Heartbeat thread started.")
        return True

    def cleanup_redis(self):
        print("Cleaning up Redis streams...")
        try:
            reset_redis = os.getenv("RESET_REDIS", "0") == "1"
            if reset_redis:
                streams = [VEHICLE_JOBS_STREAM, VEHICLE_RESULTS_STREAM, VEHICLE_ACK_STREAM]
                for stream in streams:
                    try:
                        self.r.delete(stream)
                    except Exception:
                        pass
            
            consumer_groups = {
                VEHICLE_JOBS_STREAM: [OCR_GROUP, COLOR_GROUP, LOGO_GROUP, VIOLATION_GROUP],
                VEHICLE_RESULTS_STREAM: [AGGREGATOR_GROUP],
                VEHICLE_ACK_STREAM: [INGEST_GROUP]
            }
            
            for stream_name, groups in consumer_groups.items():
                for group in groups:
                    try:
                        self.r.xgroup_create(stream_name, group, id='0', mkstream=True)
                    except Exception as e:
                        if "BUSYGROUP" not in str(e):
                            print(f"  Error creating group '{group}': {e}")
            print("  Redis cleanup complete")
        except Exception as e:
            print(f"Redis cleanup failed: {e}")
            return False
        return True
    
    def log_reader(self, process, name, color_code):
        try:
            for line in iter(process.stdout.readline, ''):
                if line and not self.shutdown_requested:
                    timestamp = time.strftime('%H:%M:%S')
                    display_name = (name[:10] + '..') if len(name) > 12 else name
                    labeled_line = f"\033[{color_code}m[{display_name:>12}]\033[0m \033[90m{timestamp}\033[0m | {line.rstrip()}"
                    print(labeled_line)
        except Exception:
            pass
    
    def start_process(self, name, command, color_code, cwd=None, extra_env=None):
        if name in self.processes and self.processes[name].poll() is None:
            return True 

        print(f"Starting {name}...")
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:."
            env['PYTHONUNBUFFERED'] = '1' 

            # Limiting threads for common libraries to prevent CPU contention
            # env['OMP_NUM_THREADS'] = '1' 
            # env['MKL_NUM_THREADS'] = '1' 
            # env['OPENBLAS_NUM_THREADS'] = '1' 
            # env['VECLIB_MAXIMUM_THREADS'] = '1' 
            # env['NUMEXPR_NUM_THREADS'] = '1'

            if extra_env:
                env.update(extra_env)
            
            process = subprocess.Popen(
                command,
                cwd=cwd or ".",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[name] = process
            
            log_thread = threading.Thread(
                target=self.log_reader, 
                args=(process, name, color_code),
                daemon=True
            )
            log_thread.start()
            
            return True
        except Exception as e:
            print(f"  Failed to start {name}: {e}")
            return False

    def stop_process(self, name):
        process = self.processes.get(name)
        if not process:
            return
        
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except:
                pass
        
        if name in self.processes:
            del self.processes[name]
        print(f"  Stopped worker: {name}")

    def start_ingress_streams(self):
        print("\nStarting Ingress Workers...")
        dev_mode = os.getenv("DEV_MODE", "0") == "1"
        visual_mode = os.getenv("VISUAL_MODE", "0") == "1"
        for idx, stream_url in enumerate(self.rtsp_streams):
            stream_url = stream_url.strip()
            if not stream_url: continue

            worker_id = f"Ingress-{idx+1}"
            ingress_env = {
                "LOCATION": self.location,
                "RTSP_STREAM": stream_url,
                "DEV_MODE": "1" if dev_mode else "0",
                "VISUAL_MODE": "1" if dev_mode else "0"
            }
            
            # Allow visual only for first stream if enabled in env
            if not dev_mode and idx == 0 and visual_mode:
                ingress_env["VISUAL_MODE"] = "1"

            success = self.start_process(
                worker_id,
                ["python3", "ingress/ingress.py"],
                "91",
                extra_env=ingress_env
            )
            
            if success:
                self.ingress_processes.append(worker_id)
            else:
                return False
        return True

    def start_worker_group(self, w_type, count=1):
        cfg = SCALING_CONFIG[w_type]
        for _ in range(count):
            if len(self.worker_groups[w_type]) >= cfg['max']:
                break
                
            existing_indices = [int(n.split('-')[-1]) for n in self.worker_groups[w_type]]
            next_idx = 1
            while next_idx in existing_indices:
                next_idx += 1
                
            worker_name = f"{w_type}-{next_idx}"
            if self.start_process(worker_name, ["python3", cfg['script']], cfg['color']):
                self.worker_groups[w_type].append(worker_name)
    
    def scale_down_worker(self, w_type):
        workers = self.worker_groups[w_type]
        cfg = SCALING_CONFIG[w_type]
        if len(workers) > cfg['min']:
            worker_to_remove = workers.pop()
            self.stop_process(worker_to_remove)

    def get_lag_counts(self):
        """Query Redis for LAG (backlog) per group."""
        stats = {}
        try:
            # XINFO GROUPS returns 'lag' in Redis 7+
            # 'pending' is only un-acked messages (active work)
            groups = self.r.xinfo_groups(VEHICLE_JOBS_STREAM)
            
            for g in groups:
                # Use 'lag' if available (Queue Size)
                # If None (older Redis), assume 'pending' is best guess, 
                # OR calculate roughly: stream_len - last_delivered (simplified)
                lag = g.get('lag')
                if lag is None:
                    # Fallback for older Redis: Use pending as proxy (less accurate)
                    lag = g.get('pending', 0)
                
                stats[g['name']] = lag
        except Exception:
            pass 
        return stats

    def check_and_autoscale(self):
        """Check queues and scale workers up/down"""
        # Use Lag, not Pending
        backlog = self.get_lag_counts()
        
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent
        can_scale_up = (cpu_usage < MAX_CPU_PERCENT) and (ram_usage < MAX_RAM_PERCENT)

        for w_type, cfg in SCALING_CONFIG.items():
            group_name = cfg['group']
            # Default to 0 if group not created yet
            current_lag = backlog.get(group_name, 0)
            active_workers = len(self.worker_groups[w_type])
            
            # SCALE UP
            if current_lag > cfg['threshold'] and active_workers < cfg['max']:
                if can_scale_up:
                    print(f"\n[AUTOSCALE] High Lag on {w_type} (Backlog: {current_lag}). Spawning worker...")
                    self.start_worker_group(w_type, 1)
                else:
                    print(f"\n[AUTOSCALE] Wanted to scale {w_type}, but system stressed ({cpu_usage}% CPU)")

            # SCALE DOWN (Only if queue is totally empty)
            elif current_lag == 0 and active_workers > cfg['min']:
                print(f"\n[AUTOSCALE] {w_type} queue empty. Scaling down...")
                self.scale_down_worker(w_type)

    def start_aggregator(self):
        print("\nStarting Edge Aggregator...")
        try:
            aggregator = ResultAggregator()
            agg_thread = threading.Thread(target=aggregator.process_results, daemon=True)
            agg_thread.start()
            return True
        except Exception:
            return False

    def start_monitor(self):
        return self.start_process("Monitor", ["python3", "db_redis/monitor_streams.py"], "96")

    def get_stream_snapshot(self):
        snapshot = {}
        for stream in [VEHICLE_JOBS_STREAM, VEHICLE_RESULTS_STREAM, VEHICLE_ACK_STREAM]:
            try:
                info = self.r.xinfo_stream(stream)
                groups_info = self.r.xinfo_groups(stream)
                snapshot[stream] = {
                    "messages": info.get("length", 0),
                    "groups": groups_info,
                }
            except Exception:
                snapshot[stream] = None
        return snapshot

    def monitor_system(self):
        print(f"\n{'='*80}")
        print("SENTINEL SYSTEM RUNNING - Auto-Scaling Enabled")
        print(f"{'='*80}")
        
        try:
            while True:
                # Cleanup dead processes
                dead_processes = []
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        dead_processes.append(name)
                
                for name in dead_processes:
                    print(f"\n[ALERT] Process died: {name}")
                    for w_type in self.worker_groups:
                        if name in self.worker_groups[w_type]:
                            self.worker_groups[w_type].remove(name)
                    del self.processes[name]

                # Check Ingress health
                if self.ingress_processes:
                    alive_ingress = [i for i in self.ingress_processes if i not in dead_processes]
                    if not alive_ingress:
                        print("All Ingress processes died. Shutting down.")
                        self.stop_all()
                        sys.exit(1)

                # Run Scaling
                self.check_and_autoscale()

                # per-process resource stats
                now = time.time()
                if now - self.last_resource_log_ts >= RESOURCE_LOG_INTERVAL:
                    usage_parts = []
                    for name, process in self.processes.items():
                        if process.poll() is not None:
                            continue
                        try:
                            p = psutil.Process(process.pid)
                            cpu_pct = p.cpu_percent(interval=None)
                            ram_mb = p.memory_info().rss / (1024 * 1024)
                            usage_parts.append(f"{name}:CPU {cpu_pct:.1f}% RAM {ram_mb:.0f}MB")
                        except Exception:
                            continue

                    stream_snapshot = self.get_stream_snapshot()
                    worker_counts = (
                        f"OCR: {len(self.worker_groups['ocr'])} "
                        f"COLOR: {len(self.worker_groups['color'])} "
                        f"LOGO: {len(self.worker_groups['logo'])} "
                        f"VIOLATION: {len(self.worker_groups['violation'])}"
                    )

                    print(f"\n[Monitor] {worker_counts}")
                    if usage_parts:
                        print("[RES] " + " | ".join(usage_parts))

                    print(f"[Monitor] Timestamp: {time.strftime('%H:%M:%S')}")
                    print("[Monitor] " + "-" * 30)
                    for stream_name, stream_data in stream_snapshot.items():
                        if stream_data is None:
                            print(f"[Monitor] {stream_name}: Stream does not exist")
                            continue

                        print(f"[Monitor] {stream_name}:")
                        print(f"[Monitor]   Messages: {stream_data['messages']}")
                        print(f"[Monitor]   Groups: {len(stream_data['groups'])}")

                        for group in stream_data["groups"]:
                            group_name = group.get("name", "unknown")
                            pending = group.get("pending", 0)
                            consumers = group.get("consumers", 0)
                            print(
                                f"[Monitor]     Group '{group_name}': {pending} pending, {consumers} consumers"
                            )

                    update_redis_metrics(stream_snapshot)

                    self.last_resource_log_ts = now

                # Status Line
                alive_count = len([p for p in self.processes.values() if p.poll() is None])
                status_str = f"\033[90m[STATUS]\033[0m Alive: {alive_count} | "
                for w_type, workers in self.worker_groups.items():
                    status_str += f"{w_type.upper()}: {len(workers)} "
                
                print(status_str, end='\r')
                time.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n\nShutdown requested...")
            self.stop_all()
            sys.exit(0)
    
    def stop_all(self):
        with self.shutdown_lock:
            if self.shutdown_requested: return
            self.shutdown_requested = True

        print(f"\n{'='*50}")
        print("Stopping all processes...")
        for name in list(self.processes.keys()):
            self.stop_process(name)
        print("All processes stopped")

    def run(self):
        print("SENTINEL SYSTEM ORCHESTRATOR")
        print("=" * 80)

        start_metrics_pusher()

        if not self.start_heartbeat(): return False
        if not self.cleanup_redis(): return False
        
        if not self.start_ingress_streams():
            self.stop_all()
            return False
            
        print("\nBootstrapping Workers...")
        for w_type in SCALING_CONFIG:
            self.start_worker_group(w_type, count=SCALING_CONFIG[w_type]['min'])
        
        time.sleep(2)
        if not self.start_aggregator():
            self.stop_all()
            return False
        
        self.monitor_system()
        self.stop_all()
        return True

def signal_handler(sig, frame):
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    orchestrator = SentinelOrchestrator()
    try:
        orchestrator.run()
    except Exception as e:
        print(f"Orchestrator failed: {e}")
        orchestrator.stop_all()
        sys.exit(1)