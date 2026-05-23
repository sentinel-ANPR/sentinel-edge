#!/usr/bin/env python3
"""Clear all messages from a Redis stream while preserving consumer groups.

This script ACKs pending messages for all consumer groups on the stream
and deletes all messages in batches. It does not delete consumer groups.

Usage:
  python3 clear_stream_data.py --stream vehicle_jobs

Use --yes to skip confirmation.
"""
import sys
import time
import argparse
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from db_redis.sentinel_redis_config import get_redis_connection


def confirm(prompt):
    resp = input(f"{prompt} [y/N]: ")
    return resp.strip().lower() in ("y", "yes")


def safe_xinfo_groups(r, stream):
    try:
        return r.xinfo_groups(stream)
    except Exception:
        try:
            return r.execute_command("XINFO", "GROUPS", stream) or []
        except Exception:
            return []


def safe_xpending_range(r, stream, group, count):
    # Try redis-py helper, fall back to execute_command
    try:
        return r.xpending_range(stream, group, min='-', max='+', count=count)
    except Exception:
        try:
            return r.execute_command("XPENDING", stream, group, "-", "+", count) or []
        except Exception:
            return []


def clear_stream(stream, batch=1000, yes=False):
    r = get_redis_connection()

    if not yes:
        print(f"About to clear all messages from stream: {stream}")
        print("This will ACK pending entries and delete all messages in the stream.")
        if not confirm("Continue?"):
            print("Aborted by user.")
            return

    # 1) Acknowledge and delete pending entries for all consumer groups
    groups = safe_xinfo_groups(r, stream)
    if not groups:
        print(f"No consumer groups found for stream '{stream}' or stream missing.")
    else:
        print(f"Found {len(groups)} groups on stream '{stream}' - processing pending entries...")
        for g in groups:
            # redis-py may return dicts or lists; try to extract group name
            try:
                group_name = g.get('name') if isinstance(g, dict) else g[1]
            except Exception:
                group_name = None
            if not group_name:
                continue
            print(f" - Group: {group_name}")
            while True:
                pending = safe_xpending_range(r, stream, group_name, batch)
                if not pending:
                    break
                ids = []
                for entry in pending:
                    # entry may be tuple/list where id is first element, or dict
                    if isinstance(entry, (list, tuple)):
                        ids.append(entry[0])
                    elif isinstance(entry, dict) and 'message_id' in entry:
                        ids.append(entry['message_id'])
                if not ids:
                    break
                # Acknowledge and delete
                try:
                    r.xack(stream, group_name, *ids)
                except Exception:
                    pass
                try:
                    for _id in ids:
                        r.xdel(stream, _id)
                except Exception:
                    pass
                print(f"   acked & deleted {len(ids)} pending entries for group {group_name}")

    # 2) Delete any remaining messages in the stream
    print(f"Deleting remaining messages from stream '{stream}' in batches of {batch}...")
    total_deleted = 0
    while True:
        try:
            msgs = r.xrange(stream, min='-', max='+', count=batch)
        except Exception:
            try:
                msgs = r.execute_command('XRANGE', stream, '-', '+', 'COUNT', batch) or []
            except Exception:
                msgs = []
        if not msgs:
            break
        ids = [m[0] if isinstance(m, (list, tuple)) else m for m in msgs]
        for _id in ids:
            try:
                r.xdel(stream, _id)
                total_deleted += 1
            except Exception:
                pass
        print(f"  deleted batch of {len(ids)} messages (total deleted: {total_deleted})")
        # small sleep to avoid blocking Redis too long
        time.sleep(0.01)

    print(f"Done. Total messages deleted from '{stream}': {total_deleted}")
    try:
        print(f"Stream length now: {r.xlen(stream)}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Clear Redis stream data but preserve groups")
    parser.add_argument("--stream", default="vehicle_jobs", help="Stream name to clear")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size for deletions")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    clear_stream(args.stream, batch=args.batch, yes=args.yes)


if __name__ == '__main__':
    main()
