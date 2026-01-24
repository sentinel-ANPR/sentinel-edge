import redis
from db_redis.sentinel_redis_config import REDIS_HOST, REDIS_PORT, REDIS_DB

# The streams defined in your setup_sentinel_redis.py
STREAMS = ["vehicle_jobs", "vehicle_results", "vehicle_ack"]

def flush_sentinel_streams():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    
    try:
        r.ping()
        for stream in STREAMS:
            if r.exists(stream):
                # Deleting the key removes the stream and all associated consumer groups
                r.delete(stream)
                print(f"  ✓ Flushed stream: {stream}")
            else:
                print(f"  - Stream '{stream}' does not exist. Skipping.")
        
        print("\nRedis Sentinel streams cleared successfully.")
        print("Note: Run 'python3 -m db_redis.setup_sentinel_redis' to recreate groups.")
        
    except redis.ConnectionError:
        print("Error: Could not connect to Redis. Is the server running?")

if __name__ == "__main__":
    confirm = input("This will delete all Sentinel Redis streams and groups. Proceed? (y/N): ")
    if confirm.lower() == 'y':
        flush_sentinel_streams()
    else:
        print("Operation cancelled.")
