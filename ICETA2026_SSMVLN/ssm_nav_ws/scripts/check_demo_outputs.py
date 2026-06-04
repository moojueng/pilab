#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
GAZEBO_DIR = WORKSPACE / "results/gazebo_rgbd"
RGBD_AGGREGATE = WORKSPACE / "results/voxel_sim/rgbd_frontier/aggregate.csv"


def status(ok, label, detail=""):
    marker = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[{marker}] {label}{suffix}")
    return ok


def read_first_row(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            return reader.fieldnames or [], row
        return reader.fieldnames or [], {}


def check_file(path, required=True):
    exists = path.exists() and path.stat().st_size > 0
    return status(
        exists or not required,
        str(path.relative_to(WORKSPACE)),
        f"{path.stat().st_size} bytes" if path.exists() else "missing",
    )


def check_voxels():
    path = GAZEBO_DIR / "observed_voxels.csv"
    if not check_file(path):
        return False
    free = 0
    occupied = 0
    total = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = int(row.get("value", -1))
            if value == 0:
                free += 1
            elif value == 1:
                occupied += 1
            total += 1
    return status(total > 0 and free > 0, "observed voxel content", f"total={total} free={free} occupied={occupied}")


def check_metrics():
    path = GAZEBO_DIR / "metrics.csv"
    if not check_file(path):
        return False
    fields, row = read_first_row(path)
    required = {
        "mission_mode",
        "coverage_ratio",
        "target_event_count",
        "confirmed_target_count",
        "frontier_exhausted",
    }
    missing = sorted(required - set(fields))
    if missing:
        return status(False, "metrics schema", f"old or incomplete schema, missing={','.join(missing)}")
    detail = (
        f"mission={row.get('mission_mode')} "
        f"coverage={row.get('coverage_ratio')} "
        f"events={row.get('target_event_count')} "
        f"confirmed={row.get('confirmed_target_count')}"
    )
    return status(True, "metrics schema", detail)


def check_rgbd_aggregate():
    if not check_file(RGBD_AGGREGATE):
        return False
    with open(RGBD_AGGREGATE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return status(False, "RGB-D aggregate", "no rows")
    detail = "; ".join(
        f"{row.get('mode')}: success={row.get('success_rate')} steps={row.get('avg_steps_success')}"
        for row in rows
    )
    return status(True, "RGB-D aggregate", detail)


def check_ros_topics():
    required = {
        "/camera/image_raw",
        "/camera/depth/image_raw",
        "/camera/depth/camera_info",
        "/odom",
    }
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source install/setup.bash && "
        "ros2 topic list"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=WORKSPACE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return status(False, "ROS topics", "ros2 topic list timed out")
    topics = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    missing = sorted(required - topics)
    return status(not missing, "ROS topics", "missing=" + ",".join(missing) if missing else "all required topics visible")


def main():
    parser = argparse.ArgumentParser(description="Check demo outputs before the professor meeting.")
    parser.add_argument("--ros-topics", action="store_true", help="Also check live ROS topics.")
    args = parser.parse_args()

    checks = [
        check_rgbd_aggregate(),
        check_voxels(),
        check_metrics(),
        check_file(GAZEBO_DIR / "trajectory.csv"),
        check_file(GAZEBO_DIR / "frontier_features.csv", required=False),
        check_file(GAZEBO_DIR / "runtime_graph_nodes.csv", required=False),
        check_file(GAZEBO_DIR / "target_events.csv", required=False),
    ]
    if args.ros_topics:
        checks.append(check_ros_topics())

    if all(checks):
        print("\nDemo outputs look ready.")
    else:
        print("\nSome checks need attention before the meeting.")


if __name__ == "__main__":
    main()
