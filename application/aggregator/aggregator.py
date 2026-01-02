import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Set timezone to IST
os.environ["TZ"] = "Asia/Kolkata"
if sys.platform != "win32":
    time.tzset()

from application.aggregator.modules.aggregator_engine import ResultAggregator

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