from pathlib import Path
import csv
from rosbags.highlevel import AnyReader

def parse_bag_to_csv(bag_path: str, target_topic: str, csv_save: str):
    bag = Path(bag_path)
    with AnyReader([bag]) as reader:
        connections = list(reader.connections)

        print("==== Topic List ====")
        for conn in connections:
            print(f"{conn.topic:<50} {conn.msgtype}")
        print("====================\n")

        selected_conns = [c for c in connections if c.topic == target_topic]
        if not selected_conns:
            raise RuntimeError(f"找不到话题 {target_topic}")

        with open(csv_save, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            # PointStamped 只有时间+xyz，无姿态
            writer.writerow(["x", "y", "z","time_sec"])
            counter = 0

            for conn, timestamp_ns, raw_bytes in reader.messages(connections=selected_conns):
                msg = reader.deserialize(raw_bytes, conn.msgtype)

                # =====适配 PointStamped=====
                pos = msg.point
                t = timestamp_ns / 1e9

                writer.writerow([ pos.x, pos.y, pos.z, t ])
                counter += 1

    print(f"处理完成，一共 {counter} 条数据，输出：{csv_save}")


if __name__ == "__main__":
    # =========修正话题名称=========
    BAG_PATH = r'C:\Users\86134\Desktop\drone\data\raw\mmaud\v1_rooftop_simple\dji_mavic3\2023-08-24-11-14-40_mavic3_gt.bag'
    TARGET_TOPIC = "/leica/point/relative"
    OUTPUT_CSV = "leica_groundtruth.csv"

    parse_bag_to_csv(BAG_PATH, TARGET_TOPIC, OUTPUT_CSV)