import time
import psutil
import sys
import os

CHECK_INTERVAL = 15

# mapping friendly names to unique strings in their command lines
PROCESS_MAPPING = {
    "Ingress": "ingress/ingress.py",
    "OCR": "ocr/ocr_worker.py",
    "Color": "color_detection/color_worker.py",
    "Logo": "logo_detection/logo_worker.py",
    "Violation": "violation_detection/violation_worker.py",
    "Orchestrator": "sentinel_orchestrator.py"
}

def get_process_stats(prev_stats):
    current_stats = {}
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline: continue
            
            cmd_str = " ".join(cmdline)
            
            for name, script_signature in PROCESS_MAPPING.items():
                if script_signature in cmd_str:
                    # using oneshot is faster/efficient
                    with proc.oneshot():
                        cpu = proc.cpu_percent(interval=None)
                        mem_bytes = proc.memory_info().rss
                        mem_mb = mem_bytes / (1024 * 1024)
                    
                    current_stats[name] = {
                        "pid": proc.info['pid'],
                        "cpu": cpu,
                        "mem": mem_mb
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
            
    return current_stats

def format_delta(current, previous, key, unit="", show_sign=False):
    if not previous:
        return ""
    
    delta = current - previous
    # only show delta if it's significant (e.g. > 0.1)
    if abs(delta) < 0.1:
        return ""
        
    sign = "+" if delta > 0 else ""
    return f"({sign}{delta:.1f}{unit})"

if __name__ == "__main__":
    print("Resource Monitor initialized.")
    sys.stdout.flush()
    
    # first run to prime CPU counters (psutil returns 0.0 on first call)
    prev_stats = get_process_stats(None)
    time.sleep(1) 
    
    try:
        while True:
            stats = get_process_stats(prev_stats)
    
            log_entries = []
            for name, data in stats.items():
                cpu = data['cpu']
                mem = data['mem']
                
                # get previous data for deltas
                prev_cpu = prev_stats.get(name, {}).get('cpu', 0)
                prev_mem = prev_stats.get(name, {}).get('mem', 0)
                
                cpu_delta = format_delta(cpu, prev_cpu, 'cpu', "%")
                mem_delta = format_delta(mem, prev_mem, 'mem', "")
                
                # format: "Ingress: 45% CPU (+2%) | 200MB Mem (+10)"
                entry = f"{name}: {cpu:>4.1f}% CPU {cpu_delta} | {mem:>5.1f}MB Mem {mem_delta}"
                print(entry)
            
            # flush stdout so Orchestrator sees it immediately
            sys.stdout.flush()
            
            prev_stats = stats
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        pass