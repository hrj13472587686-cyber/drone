from pathlib import Path
import csv, math, argparse
from rosbags.highlevel import AnyReader

FIELDNAMES = ["timestamp","frame_id","object_id","source","x","y","z","vx","vy","vz","confidence","split","meta"]

def extract_pointmsgs_from_bag(bag_path: str, topic: str):
    bag = Path(bag_path)
    with AnyReader([bag]) as reader:
        conns = list(reader.connections)
        selected = [c for c in conns if c.topic == topic]
        if not selected:
            return []  # no messages for this topic
        rows = []
        for conn, timestamp_ns, raw in reader.messages(connections=selected):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                p = msg.point
                t = float(timestamp_ns) / 1e9
                x, y, z = float(p.x), float(p.y), float(p.z)
                rows.append((t, x, y, z))
            except Exception:
                # skip malformed messages
                continue
    return rows

def make_unified(bag_paths, topic, out_csv):
    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fid = 0
    written = 0
    with out_p.open("w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=FIELDNAMES)
        writer.writeheader()
        for bag in bag_paths:
            print(f"Processing bag: {bag}")
            rows = extract_pointmsgs_from_bag(bag, topic)
            print(f"  found {len(rows)} messages on {topic}")
            for (t,x,y,z) in rows:
                if not all(map(lambda v: isinstance(v, float) and math.isfinite(v), (t,x,y,z))):
                    continue
                writer.writerow({
                    "timestamp": f"{t:.9f}",
                    "frame_id": str(fid),
                    "object_id": "mavic3",
                    "source": "gt",
                    "x": f"{x}",
                    "y": f"{y}",
                    "z": f"{z}",
                    "vx": "",
                    "vy": "",
                    "vz": "",
                    "confidence": "1.0",
                    "split": "week1",
                    "meta": f"mmaud;topic={topic}"
                })
                fid += 1
                written += 1
    print(f"Wrote {written} records to {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Combine bags into unified GT CSV")
    p.add_argument("--bags", nargs="+", required=False, default=[
        r"C:\Users\86134\Desktop\drone\data\raw\mmaud\v1_rooftop_simple\dji_mavic3\2023-08-24-11-14-40_mavic3_gt.bag",
        r"C:\Users\86134\Desktop\drone\data\raw\mmaud\v1_rooftop_simple\dji_mavic3\Mavic3.bag",
    ], help="One or more .bag files")
    p.add_argument("--topic", default="/leica/point/relative", help="PointStamped topic to extract")
    p.add_argument("--out", default=r"data\trajectories\mmaud_mavic3_gt_relative.csv", help="Output unified CSV")
    args = p.parse_args()

    make_unified(args.bags, args.topic, args.out)