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
    """
    Entry point to initialize and start the ResultAggregator loop.
    This version is designed for the Edge Node (Uploader Mode).
    """
    print(f"\n--- Starting Sentinel Edge Aggregator ---")
    print(f"Node ID: {os.getenv('NODE_ID')}")
    print(f"Location: {os.getenv('LOCATION')}")
    print(f"Central Server: {os.getenv('CENTRAL_API_URL')}")
    
    # 1. Initialize the Aggregator Engine
    # Note: On the Edge, we don't need manager, loop, or DB connection
    aggregator = ResultAggregator()

    try:
        # 2. Start the blocking process loop
        # This polls local Redis and POSTS results + images to Central
        aggregator.process_results()
        
    except KeyboardInterrupt:
        print("\nEdge Aggregator shutting down...")
    except Exception as e:
        print(f"Critical error in Edge Aggregator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_edge_aggregator()