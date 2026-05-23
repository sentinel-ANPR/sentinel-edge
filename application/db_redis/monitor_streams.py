import redis
import time
import os
from db_redis.sentinel_redis_config import *

MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_STREAMS_INTERVAL", "15"))

def monitor_streams():
    """Monitor all Redis streams"""
    r = get_redis_connection()
    
    print("Sentinel Redis Stream Monitor")
    print("=" * 50)
    
    while True:
        try:
            print(f"\nTimestamp: {time.strftime('%H:%M:%S')}")
            print("-" * 30)
            
            # Monitor each stream
            for stream in [
                VEHICLE_JOBS_STREAM,
                VEHICLE_RESULTS_STREAM,
                VEHICLE_RESULTS_DEAD_STREAM,
                VEHICLE_ACK_STREAM,
            ]:
                try:
                    info = r.xinfo_stream(stream)
                    try:
                        groups_info = r.xinfo_groups(stream)
                    except Exception:
                        groups_info = []
                    
                    print(f"{stream}:")
                    print(f"  Messages: {info['length']}")
                    print(f"  Groups: {len(groups_info)}")
                    
                    for group in groups_info:
                        pending = group['pending']
                        consumers = group['consumers']
                        print(f"    Group '{group['name']}': {pending} pending, {consumers} consumers")
                    
                except redis.ResponseError:
                    print(f"{stream}: Stream does not exist")
            
            time.sleep(MONITOR_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_streams()