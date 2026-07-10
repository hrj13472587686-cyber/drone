from pathlib import Path
import csv
import argparse
from collections import defaultdict
from rosbags.highlevel import AnyReader

def enumerate_bag_topics(bag_path: str, out_csv: str = None):
    bag = Path(bag_path)
    with AnyReader([bag]) as reader:
        conns = list(reader.connections)
        counts = {(c.topic, c.msgtype): 0 for c in conns}
        for conn, _, _ in reader.messages(connections=conns):
            counts[(conn.topic, conn.msgtype)] += 1

    print(f"{'topic':<60} {'msgtype':<40} {'count':>8}")
    print("-" * 110)
    for (topic, msgtype), cnt in sorted(counts.items()):
        print(f"{topic:<60} {msgtype:<40} {cnt:>8}")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["topic", "msgtype", "count"])
            for (topic, msgtype), cnt in sorted(counts.items()):
                w.writerow([topic, msgtype, cnt])
    return counts

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="List topics and message counts in a ROS bag")
    p.add_argument("--bag",  required=False,     default=r"C:\Users\86134\Desktop\drone\data\raw\mmaud\v1_rooftop_simple\dji_mavic3\2023-08-24-11-14-40_mavic3_gt.bag")
    p.add_argument("--out", help="Output CSV file to save the topic list and counts")
    args = p.parse_args()
    enumerate_bag_topics(args.bag, args.out)