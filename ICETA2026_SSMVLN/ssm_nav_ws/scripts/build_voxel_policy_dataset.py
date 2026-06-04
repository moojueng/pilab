#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from voxel_nav_common import ACTIONS_3D, VoxelPos, action_from_to, bfs_path, find_target, free_like, in_bounds, load_voxel_grid, neighbors6, shape3


PATCH_RADIUS = 1
PATCH_SIZE = (2 * PATCH_RADIUS + 1) ** 3


def start_for(grid):
    _, rows, _ = shape3(grid)
    return VoxelPos(0, rows - 1, 0)


def local_patch(grid, pos, radius=PATCH_RADIUS):
    vals = []
    for dz in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                p = VoxelPos(pos.z + dz, pos.r + dr, pos.c + dc)
                vals.append(float(grid[p.z][p.r][p.c]) if in_bounds(grid, p) else -1.0)
    return vals


def observe(grid, observed, pos, radius=PATCH_RADIUS):
    for dz in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                p = VoxelPos(pos.z + dz, pos.r + dr, pos.c + dc)
                if in_bounds(grid, p):
                    observed[p] = grid[p.z][p.r][p.c]


def graph_summary(grid, observed, visited):
    nodes = [p for p, v in observed.items() if v != 1]
    node_set = set(nodes)
    edges = 0
    frontier = 0
    for p in nodes:
        for _, n in neighbors6(grid, p):
            if n in node_set:
                edges += 1
            if n not in observed:
                frontier += 1
    return len(nodes), edges, frontier, len(visited)


def action_reward(grid, observed, visited, pos, action):
    dz, dr, dc = ACTIONS_3D[action]
    nxt = VoxelPos(pos.z + dz, pos.r + dr, pos.c + dc)
    if action == 6:
        return 10.0 if 2.0 in local_patch(grid, pos) else -0.5
    if not free_like(grid, nxt):
        return -3.0
    reward = -0.03
    reward += 0.6 if nxt not in visited else -0.45
    if nxt not in observed:
        reward += 0.35
    for _, nn in neighbors6(grid, nxt):
        if nn not in observed:
            reward += 0.18
    if grid[nxt.z][nxt.r][nxt.c] == 2:
        reward += 10.0
    if action in (4, 5):
        reward -= 0.05
    return reward


def build_rows_for_map(map_path):
    grid = load_voxel_grid(map_path)
    start = start_for(grid)
    target = find_target(grid)
    path = bfs_path(grid, start, target)
    if len(path) < 2:
        return []

    observed = {}
    visited = set()
    prev_action = 6
    rows = []
    for step, pos in enumerate(path[:-1]):
        observe(grid, observed, pos)
        nodes, edges, frontier, visited_count = graph_summary(grid, observed, visited)
        teacher_action = action_from_to(pos, path[step + 1])
        rewards = [action_reward(grid, observed, visited, pos, a) for a in range(7)]
        best_reward_action = max(range(7), key=lambda a: rewards[a])
        row = {
            "map_name": Path(map_path).stem,
            "step": step,
            "robot_z": pos.z,
            "robot_r": pos.r,
            "robot_c": pos.c,
            "prev_action": prev_action,
            "observed_nodes": nodes,
            "observed_edges": edges,
            "frontier_count": frontier,
            "visited_count": visited_count,
            "teacher_action": teacher_action,
            "best_reward_action": best_reward_action,
        }
        for i, v in enumerate(rewards):
            row[f"reward_{i}"] = v
        for i, v in enumerate(local_patch(grid, pos)):
            row[f"patch_{i}"] = v
        rows.append(row)
        visited.add(pos)
        prev_action = teacher_action
    return rows


def write_dataset(map_dir, out_csv):
    map_files = sorted(Path(map_dir).glob("*.vxl"))
    all_rows = []
    for p in map_files:
        all_rows.extend(build_rows_for_map(p))
    if not all_rows:
        raise RuntimeError(f"no rows generated from {map_dir}")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved {out_csv}, rows={len(all_rows)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    write_dataset(args.map_dir, args.out)


if __name__ == "__main__":
    main()
