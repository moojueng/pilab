#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from pathlib import Path

ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 0),   # stop
}

def load_grid(path):
    grid = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                grid.append([int(x) for x in line.split(",")])
    return grid

def find_target(grid):
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == 2:
                return (r, c)
    raise ValueError("target cell 2 not found")

def in_bounds(grid, p):
    r, c = p
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])

def free_like(grid, p):
    r, c = p
    return in_bounds(grid, p) and grid[r][c] != 1

def neighbors4(grid, p):
    r, c = p
    for a in [0, 1, 2, 3]:
        dr, dc = ACTIONS[a]
        n = (r + dr, c + dc)
        if free_like(grid, n):
            yield a, n

def astar_bfs_teacher(grid, start, target):
    q = deque([start])
    parent = {start: None}
    parent_action = {}

    while q:
        cur = q.popleft()
        if cur == target:
            break
        for a, n in neighbors4(grid, cur):
            if n not in parent:
                parent[n] = cur
                parent_action[n] = a
                q.append(n)

    if target not in parent:
        return []

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def local_patch(grid, pos, radius=2):
    r0, c0 = pos
    vals = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            p = (r0 + dr, c0 + dc)
            if not in_bounds(grid, p):
                vals.append(-1)
            else:
                vals.append(grid[p[0]][p[1]])
    return vals

def observe(grid, observed, pos, radius=2):
    r0, c0 = pos
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            p = (r0 + dr, c0 + dc)
            if in_bounds(grid, p):
                observed[p] = grid[p[0]][p[1]]

def graph_summary(grid, observed, visited):
    nodes = [p for p, v in observed.items() if v != 1]
    node_set = set(nodes)
    edges = 0
    frontier = 0

    for p in nodes:
        for _, n in neighbors4(grid, p):
            if n in node_set:
                edges += 1
            if n not in observed:
                frontier += 1

    return len(nodes), edges, frontier, len(visited)

def action_from_to(a, b):
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for action, delta in ACTIONS.items():
        if delta == (dr, dc):
            return action
    return 4

def action_reward(grid, observed, visited, pos, action):
    dr, dc = ACTIONS[action]
    nxt = (pos[0] + dr, pos[1] + dc)

    if action == 4:
        patch = local_patch(grid, pos, 2)
        return 10.0 if 2 in patch else -0.5

    if not free_like(grid, nxt):
        return -3.0

    reward = -0.02

    if nxt not in visited:
        reward += 0.5
    else:
        reward -= 0.4

    if nxt not in observed:
        reward += 0.3

    for _, nn in neighbors4(grid, nxt):
        if nn not in observed:
            reward += 0.2

    if grid[nxt[0]][nxt[1]] == 2:
        reward += 10.0

    return reward

def build_rows_for_map(map_path, map_name, patch_radius=2):
    grid = load_grid(map_path)
    start = (len(grid) - 1, 0)
    target = find_target(grid)
    path = astar_bfs_teacher(grid, start, target)

    if len(path) < 2:
        return []

    observed = {}
    visited = set()
    rows = []
    prev_action = 4

    for step, pos in enumerate(path[:-1]):
        observe(grid, observed, pos, patch_radius)
        patch = local_patch(grid, pos, patch_radius)
        nodes, edges, frontier, visited_count = graph_summary(grid, observed, visited)

        teacher_action = action_from_to(pos, path[step + 1])
        rewards = [action_reward(grid, observed, visited, pos, a) for a in range(5)]
        best_reward_action = max(range(5), key=lambda a: rewards[a])

        row = {
            "map_name": map_name,
            "step": step,
            "robot_r": pos[0],
            "robot_c": pos[1],
            "prev_action": prev_action,
            "observed_nodes": nodes,
            "observed_edges": edges,
            "frontier_count": frontier,
            "visited_count": visited_count,
            "teacher_action": teacher_action,
            "best_reward_action": best_reward_action,
            "reward_forward": rewards[0],
            "reward_right": rewards[1],
            "reward_down": rewards[2],
            "reward_left": rewards[3],
            "reward_stop": rewards[4],
        }

        for i, v in enumerate(patch):
            row[f"patch_{i}"] = v

        rows.append(row)
        visited.add(pos)
        prev_action = teacher_action

    return rows

def write_dataset(map_dir, out_csv):
    map_files = sorted(Path(map_dir).glob("*.csv"))
    all_rows = []

    for p in map_files:
        all_rows.extend(build_rows_for_map(p, p.stem))

    if not all_rows:
        raise RuntimeError(f"no rows generated from {map_dir}")

    fieldnames = list(all_rows[0].keys())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"saved {out_csv}, rows={len(all_rows)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_dataset(args.map_dir, args.out)

if __name__ == "__main__":
    main()
