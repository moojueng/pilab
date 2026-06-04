import argparse
import csv
import heapq
import os
import random


ACTION_TO_ID = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}


def neighbors(x, y, w, h):
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def make_grid(w, h, obstacle_prob):
    return [
        [1 if random.random() < obstacle_prob else 0 for _ in range(w)]
        for _ in range(h)
    ]


def free_cells(grid):
    cells = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value == 0:
                cells.append((x, y))
    return cells


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    h = len(grid)
    w = len(grid[0])

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    parent = {start: None}
    g_score = {start: 0}

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        for nx, ny in neighbors(current[0], current[1], w, h):
            if grid[ny][nx] != 0:
                continue

            nxt = (nx, ny)
            new_cost = cost + 1

            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                parent[nxt] = current
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, new_cost, nxt))

    return []


def action_from_step(current, nxt):
    cx, cy = current
    nx, ny = nxt

    if nx == cx and ny == cy - 1:
        return ACTION_TO_ID["up"]
    if nx == cx and ny == cy + 1:
        return ACTION_TO_ID["down"]
    if nx == cx - 1 and ny == cy:
        return ACTION_TO_ID["left"]
    if nx == cx + 1 and ny == cy:
        return ACTION_TO_ID["right"]

    raise ValueError(f"Invalid step: current={current}, next={nxt}")


def local_obstacle_ratio(grid, x, y, radius=1):
    h = len(grid)
    w = len(grid[0])

    total = 0
    occupied = 0

    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            total += 1

            if xx < 0 or xx >= w or yy < 0 or yy >= h:
                occupied += 1
            elif grid[yy][xx] != 0:
                occupied += 1

    return occupied / max(total, 1)


def free_direction_features(grid, x, y):
    h = len(grid)
    w = len(grid[0])

    feats = []
    for nx, ny in [
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    ]:
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            feats.append(0.0)
        elif grid[ny][nx] == 0:
            feats.append(1.0)
        else:
            feats.append(0.0)

    return feats


def synth_vision_features(grid, x, y):
    obs = local_obstacle_ratio(grid, x, y, radius=1)

    vision_dark = min(max(obs + random.uniform(-0.05, 0.05), 0.0), 1.0)
    vision_mean = min(max(1.0 - vision_dark + random.uniform(-0.03, 0.03), 0.0), 1.0)
    vision_edge = min(max(0.2 + 0.8 * obs + random.uniform(-0.08, 0.08), 0.0), 1.0)

    return vision_mean, vision_dark, vision_edge


def make_row(grid, current, nxt, goal, path_remaining, w, h, map_name):
    cx, cy = current
    gx, gy = goal

    vision_mean, vision_dark, vision_edge = synth_vision_features(grid, cx, cy)
    free_up, free_down, free_left, free_right = free_direction_features(grid, cx, cy)

    dx_goal = (gx - cx) / max(w - 1, 1)
    dy_goal = (gy - cy) / max(h - 1, 1)

    teacher_action = action_from_step(current, nxt)

    return {
        "map_name": map_name,
        "current_x": cx / max(w - 1, 1),
        "current_y": cy / max(h - 1, 1),
        "goal_x": gx / max(w - 1, 1),
        "goal_y": gy / max(h - 1, 1),
        "delta_goal_x": dx_goal,
        "delta_goal_y": dy_goal,
        "path_remaining": path_remaining / max(w + h, 1),
        "vision_mean": vision_mean,
        "vision_dark": vision_dark,
        "vision_edge": vision_edge,
        "free_up": free_up,
        "free_down": free_down,
        "free_left": free_left,
        "free_right": free_right,
        "teacher_action": teacher_action,
    }


def generate_split(csv_path, num_maps, w, h, obstacle_prob, samples_per_map):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []

    for map_idx in range(num_maps):
        for _ in range(300):
            grid = make_grid(w, h, obstacle_prob)
            cells = free_cells(grid)

            if len(cells) < 2:
                continue

            start = random.choice(cells)
            goal = random.choice(cells)

            if start == goal:
                continue

            path = astar(grid, start, goal)

            if len(path) >= 3:
                break
        else:
            continue

        max_samples = min(samples_per_map, len(path) - 1)
        sample_indices = sorted(random.sample(range(len(path) - 1), max_samples))

        for idx in sample_indices:
            current = path[idx]
            nxt = path[idx + 1]
            path_remaining = len(path) - idx

            rows.append(
                make_row(
                    grid=grid,
                    current=current,
                    nxt=nxt,
                    goal=goal,
                    path_remaining=path_remaining,
                    w=w,
                    h=h,
                    map_name=f"synthetic_map_{map_idx:05d}",
                )
            )

    fieldnames = [
        "map_name",
        "current_x",
        "current_y",
        "goal_x",
        "goal_y",
        "delta_goal_x",
        "delta_goal_y",
        "path_remaining",
        "vision_mean",
        "vision_dark",
        "vision_edge",
        "free_up",
        "free_down",
        "free_left",
        "free_right",
        "teacher_action",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} samples -> {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/ssm_nav_actions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--obstacle-prob", type=float, default=0.25)
    parser.add_argument("--train-maps", type=int, default=1200)
    parser.add_argument("--test-maps", type=int, default=200)
    parser.add_argument("--unseen-maps", type=int, default=200)
    parser.add_argument("--samples-per-map", type=int, default=12)
    args = parser.parse_args()

    random.seed(args.seed)

    generate_split(
        os.path.join(args.out, "train.csv"),
        args.train_maps,
        args.width,
        args.height,
        args.obstacle_prob,
        args.samples_per_map,
    )

    generate_split(
        os.path.join(args.out, "test.csv"),
        args.test_maps,
        args.width,
        args.height,
        args.obstacle_prob,
        args.samples_per_map,
    )

    generate_split(
        os.path.join(args.out, "unseen.csv"),
        args.unseen_maps,
        args.width,
        args.height,
        args.obstacle_prob,
        args.samples_per_map,
    )


if __name__ == "__main__":
    main()
