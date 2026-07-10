#!/usr/bin/env python3
import csv, math, statistics, json, argparse, sys

def safe_float(s):
    try:
        v = float(s)
        return v
    except:
        return None

def compute(path):
    total = 0
    missing = 0
    parse_errors = 0
    nonfinite = 0
    ts = []; xs = []; ys = []; zs = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            t_s = (row.get('timestamp') or '').strip()
            x_s = (row.get('x') or '').strip()
            y_s = (row.get('y') or '').strip()
            z_s = (row.get('z') or '').strip()
            if t_s == '' or x_s == '' or y_s == '' or z_s == '':
                missing += 1
                # still attempt parse if present
            t = safe_float(t_s); x = safe_float(x_s); y = safe_float(y_s); z = safe_float(z_s)
            if t is None or x is None or y is None or z is None:
                parse_errors += 1
                continue
            if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                nonfinite += 1
                continue
            ts.append(t); xs.append(x); ys.append(y); zs.append(z)

    n = len(ts)
    result = {
        'total_rows': total,
        'rows_with_all_xyz_timestamp': n,
        'missing_fields_count': missing,
        'parse_errors_count': parse_errors,
        'nonfinite_count': nonfinite,
    }
    if n >= 1:
        result['first_timestamp'] = ts[0]
        result['last_timestamp'] = ts[-1]
        result['time_span_s'] = ts[-1] - ts[0] if n >= 2 else 0.0
    else:
        result['first_timestamp'] = result['last_timestamp'] = None
        result['time_span_s'] = 0.0

    if n >= 2:
        intervals = [ts[i+1] - ts[i] for i in range(n-1)]
        result['median_interval_s'] = statistics.median(intervals)
        result['mean_interval_s'] = statistics.mean(intervals)
        result['timestamp_decrease_count'] = sum(1 for d in intervals if d < 0)
    else:
        result['median_interval_s'] = None
        result['mean_interval_s'] = None
        result['timestamp_decrease_count'] = 0

    if xs:
        result.update({
            'min_x': min(xs), 'max_x': max(xs),
            'min_y': min(ys), 'max_y': max(ys),
            'min_z': min(zs), 'max_z': max(zs),
        })
    else:
        result.update({'min_x': None,'max_x': None,'min_y': None,'max_y': None,'min_z': None,'max_z': None})

    return result

def main():
    p = argparse.ArgumentParser(description='Compute GT CSV stats')
    p.add_argument('--csv', required=False,     default=r"C:\Users\86134\Desktop\drone\data\trajectories\mmaud_mavic3_gt_relative.csv")
    p.add_argument('--json-out', required=False, default=r"C:\Users\86134\Desktop\drone\data\trajectories\mmaud_mavic3_gt_relative_stats.json")
    args = p.parse_args()
    res = compute(args.csv)
    # human-readable
    print(f"total rows: {res['total_rows']}")
    print(f"valid rows (timestamp,x,y,z): {res['rows_with_all_xyz_timestamp']}")
    print(f"time span (s): {res['time_span_s']:.6f}" if res['time_span_s'] is not None else "time span: N/A")
    mi = res.get('median_interval_s')
    print(f"median interval (s): {mi:.6f}" if mi is not None else "median interval: N/A")
    print(f"x min/max: {res['min_x']} / {res['max_x']}")
    print(f"y min/max: {res['min_y']} / {res['max_y']}")
    print(f"z min/max: {res['min_z']} / {res['max_z']}")
    print(f"missing fields: {res['missing_fields_count']}, parse errors: {res['parse_errors_count']}, nonfinite: {res['nonfinite_count']}")
    print(f"timestamp decreases: {res['timestamp_decrease_count']}")
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as jf:
            json.dump(res, jf, indent=2, ensure_ascii=False)
        print(f"wrote JSON summary to {args.json_out}")

if __name__ == '__main__':
    main()