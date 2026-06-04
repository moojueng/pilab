#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import deque
from pathlib import Path


ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


def load_grid(path):
    grid = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grid.append([int(x.strip()) for x in line.split(",")])
    return grid


def free_cells(grid):
    cells = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0:
                cells.append((r, c))
    return cells


def neighbors(grid, cell):
    r, c = cell
    rows = len(grid)
    cols = len(grid[0])
    for action, (dr, dc) in ACTIONS.items():
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield action, (nr, nc)


def shortest_path(grid, start, goal):
    q = deque([start])
    parent = {start: None}
    parent_action = {}

    while q:
        cur = q.popleft()
        if cur == goal:
            break

        for action, nxt in neighbors(grid, cur):
            if nxt not in parent:
                parent[nxt] = cur
                parent_action[nxt] = action
                q.append(nxt)

    if goal not in parent:
        return None, None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    if len(path) < 2:
        return path, 0

    first_action = parent_action[path[1]]
    return path, first_action


def cell_to_world(cell, rows, cols, cell_size):
    r, c = cell
    x = -(cols - 1) * cell_size / 2.0 + c * cell_size
    y = (rows - 1) * cell_size / 2.0 - r * cell_size
    return x, y


def scan_like_features(grid, cell):
    r, c = cell
    rows = len(grid)
    cols = len(grid[0])

    def is_free(rr, cc):
        return 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == 0

    front_clear = 1.0 if is_free(r - 1, c) else 0.0
    back_clear = 1.0 if is_free(r + 1, c) else 0.0
    left_clear = 1.0 if is_free(r, c - 1) else 0.0
    right_clear = 1.0 if is_free(r, c + 1) else 0.0

    min_range = max(front_clear, back_clear, left_clear, right_clear)

    return front_clear, left_clear, right_clear, min_range


def rotate_world_to_robot(dx, dy, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    robot_x = cos_yaw * dx + sin_yaw * dy
    robot_y = -sin_yaw * dx + cos_yaw * dy
    return robot_x, robot_y


def action_to_robot_frame_label(world_action, yaw):
    ar, ac = ACTIONS[world_action]

    # grid row/col action -> world dx/dy
    # up(row-1)=+y, down(row+1)=-y, left=-x, right=+x
    if world_action == 0:
        wx, wy = 0.0, 1.0
    elif world_action == 1:
        wx, wy = 0.0, -1.0
    elif world_action == 2:
        wx, wy = -1.0, 0.0
    else:
        wx, wy = 1.0, 0.0

    rx, ry = rotate_world_to_robot(wx, wy, yaw)

    if abs(rx) >= abs(ry):
        return 0 if rx >= 0 else 1  # forward/back
    return 2 if ry >= 0 else 3      # left/right


def generate_rows(map_path, samples_per_map, cell_size, yaw_count, seed):
    random.seed(seed)

    grid = load_grid(map_path)
    rows = len(grid)
    cols = len(grid[0])
    cells = free_cells(grid)

    yaws = [2.0 * math.pi * i / yaw_count for i in range(yaw_count)]

    out_rows = []

    for _ in range(samples_per_map):
        start = random.choice(cells)
        goal = random.choice(cells)

        if start == goal:
            continue

        path, world_action = shortest_path(grid, start, goal)
        if path is None or len(path) < 2:
            continue

        sx, sy = cell_to_world(start, rows, cols, cell_size)
        gx, gy = cell_to_world(goal, rows, cols, cell_size)

        dx = gx - sx
        dy = gy - sy
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1e-6:
            continue

        world_dir_x = dx / distance
        world_dir_y = dy / distance

        front_clear, left_clear, right_clear, min_range = scan_like_features(grid, start)

        for yaw in yaws:
            robot_dir_x, robot_dir_y = rotate_world_to_robot(world_dir_x, world_dir_y, yaw)
            label = action_to_robot_frame_label(world_action, yaw)

            # synthetic vision features. Later this is replaced by real camera features.
            obstacle_ratio = 1.0 - ((front_clear + left_clear + right_clear) / 3.0)
            vision_mean = 0.65 - 0.25 * obstacle_ratio + random.uniform(-0.04, 0.04)
            vision_dark = 0.15 + 0.35 * obstacle_ratio + random.uniform(-0.03, 0.03)
            vision_edge = 0.03 + 0.15 * obstacle_ratio + random.uniform(-0.02, 0.02)

            vision_mean = min(1.0, max(0.0, vision_mean))
            vision_dark = min(1.0, max(0.0, vision_dark))
            vision_edge = min(1.0, max(0.0, vision_edge))

            out_rows.append({
                "map_path": str(map_path),
                "current_x": sx,
                "current_y": sy,
                "goal_x": gx,
                "goal_y": gy,
                "robot_dir_x": robot_dir_x,
                "robot_dir_y": robot_dir_y,
                "distance_to_goal": min(distance, 10.0),
                "vision_mean": vision_mean,
                "vision_dark": vision_dark,
                "vision_edge": vision_edge,
                "front_clear": front_clear,
                "left_clear": left_clear,
                "right_clear": right_clear,
                "min_range": min_range,
                "teacher_action": label,
            })

    return out_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "map_path",
        "current_x",
        "current_y",
        "goal_x",
        "goal_y",
        "robot_dir_x",
        "robot_dir_y",
        "distance_to_goal",
        "vision_mean",
        "vision_dark",
        "vision_edge",
        "front_clear",
        "left_clear",
        "right_clear",
        "min_range",
        "teacher_action",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {path} rows={len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-map", default="maps/train/train_map_01.csv")
    parser.add_argument("--test-map", default="maps/test/test_map_01.csv")
    parser.add_argument("--unseen-map", default="maps/unseen/unseen_map_01.csv")
    parser.add_argument("--out-dir", default="datasets/ssm_nav_robot_frame")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--yaw-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    train_rows = generate_rows(args.train_map, args.samples, args.cell_size, args.yaw_count, args.seed)
    test_rows = generate_rows(args.test_map, max(300, args.samples // 4), args.cell_size, args.yaw_count, args.seed + 1)
    unseen_rows = generate_rows(args.unseen_map, max(300, args.samples // 4), args.cell_size, args.yaw_count, args.seed + 2)

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "test.csv", test_rows)
    write_csv(out_dir / "unseen.csv", unseen_rows)


if __name__ == "__main__":
    main()
