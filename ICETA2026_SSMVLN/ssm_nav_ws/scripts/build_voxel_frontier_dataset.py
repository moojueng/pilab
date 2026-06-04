#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from pathlib import Path

from voxel_frontier_features import FRONTIER_FEATURE_NAMES, build_frontier_feature
from voxel_nav_common import ACTIONS_3D, VoxelPos, action_from_to, bfs_path, find_target, free_like, in_bounds, is_semantic_value, load_voxel_grid, shape3


def start_for(grid):
    _, rows, _ = shape3(grid)
    return VoxelPos(0, rows - 1, 0)


def neighbors6_all(grid, p):
    for dz, dr, dc in ACTIONS_3D[:6]:
        q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
        if in_bounds(grid, q):
            yield q


def observe_local(grid, observed, pos, radius):
    for dz in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                p = VoxelPos(pos.z + dz, pos.r + dr, pos.c + dc)
                if in_bounds(grid, p):
                    observed[p.z][p.r][p.c] = grid[p.z][p.r][p.c]


def observed_free(grid, observed, p):
    return in_bounds(grid, p) and (observed[p.z][p.r][p.c] == 0 or is_semantic_value(observed[p.z][p.r][p.c]))


def build_observed_graph(grid, observed):
    nodes = set()
    edges = 0
    depth, rows, cols = shape3(grid)
    for z in range(depth):
        for r in range(rows):
            for c in range(cols):
                p = VoxelPos(z, r, c)
                if observed_free(grid, observed, p):
                    nodes.add(p)
    for p in nodes:
        for n in neighbors6_all(grid, p):
            if n in nodes:
                edges += 1
    return nodes, edges


def is_frontier(grid, observed, p):
    return observed_free(grid, observed, p) and any(
        in_bounds(grid, n) and observed[n.z][n.r][n.c] == -1
        for n in neighbors6_all(grid, p)
    )


def bfs_observed(grid, observed, start):
    q = deque([start])
    parent = {start: None}
    dist = {start: 0}
    while q:
        cur = q.popleft()
        for n in neighbors6_all(grid, cur):
            if observed_free(grid, observed, n) and n not in parent:
                parent[n] = cur
                dist[n] = dist[cur] + 1
                q.append(n)
    return parent, dist


def full_distances_from_target(grid, target):
    q = deque([target])
    dist = {target: 0}
    while q:
        cur = q.popleft()
        for n in neighbors6_all(grid, cur):
            if free_like(grid, n) and n not in dist:
                dist[n] = dist[cur] + 1
                q.append(n)
    return dist


def first_action_to(robot, goal, parent):
    cur = goal
    while parent.get(cur) is not None and parent[cur] != robot:
        cur = parent[cur]
    if parent.get(cur) is None:
        return 6
    return action_from_to(robot, cur)


def unknown_count(grid, observed, p, radius):
    count = 0
    for dz in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                if in_bounds(grid, q) and observed[q.z][q.r][q.c] == -1:
                    count += 1
    return count


def hidden_free_gain(grid, observed, p, radius):
    count = 0
    for dz in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                if in_bounds(grid, q) and observed[q.z][q.r][q.c] == -1 and grid[q.z][q.r][q.c] != 1:
                    count += 1
    return count


def local_visit_penalty(grid, visits, p, radius=1):
    total = 0
    for dz in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                if in_bounds(grid, q):
                    total += min(visits[q.z][q.r][q.c], 5)
    return float(total)


def free_degree(grid, observed, p):
    return sum(1 for n in neighbors6_all(grid, p) if observed_free(grid, observed, n))


def frontier_score(grid, observed, visits, robot, frontier, graph_dist, full_target_dist, max_target_dist):
    gain = hidden_free_gain(grid, observed, frontier, 2) / 125.0
    unknown = unknown_count(grid, observed, frontier, 2) / 125.0
    target_dist = full_target_dist.get(frontier, max_target_dist)
    target_prior = 1.0 / (1.0 + float(target_dist))
    travel = min(float(graph_dist), 64.0) / 64.0
    revisit = local_visit_penalty(grid, visits, frontier, 1) / 30.0
    dead_end = 1.0 if free_degree(grid, observed, frontier) <= 1 else 0.0
    vertical = 1.0 if frontier.z != robot.z else 0.0
    return 2.2 * gain + 0.55 * unknown + 1.25 * target_prior - 0.85 * travel - 0.45 * revisit - 0.2 * dead_end - 0.12 * vertical


def build_rows_for_map(map_path, observation_radius):
    grid = load_voxel_grid(map_path)
    depth, rows, cols = shape3(grid)
    start = start_for(grid)
    target = find_target(grid)
    path = bfs_path(grid, start, target)
    if len(path) < 2:
        return []

    observed = [[[-1 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
    visits = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
    full_target_dist = full_distances_from_target(grid, target)
    max_target_dist = max(depth + rows + cols, 1)
    visited = set()
    prev_action = 6
    rows_out = []

    for step, robot in enumerate(path[:-1]):
        observe_local(grid, observed, robot, observation_radius)
        nodes, edges = build_observed_graph(grid, observed)
        parent, dist = bfs_observed(grid, observed, robot)
        frontiers = [p for p in nodes if p != robot and p in dist and is_frontier(grid, observed, p)]
        frontier_count = len(frontiers)
        if frontier_count:
            scored = [
                (p, frontier_score(grid, observed, visits, robot, p, dist[p], full_target_dist, max_target_dist))
                for p in frontiers
            ]
            best_score = max(score for _, score in scored)
            for rank, (frontier, score) in enumerate(sorted(scored, key=lambda item: item[1], reverse=True)):
                action = first_action_to(robot, frontier, parent)
                feature = build_frontier_feature(
                    robot=robot,
                    frontier=frontier,
                    shape=(depth, rows, cols),
                    graph_dist=dist[frontier],
                    first_action=action,
                    prev_action=prev_action,
                    observed_nodes=len(nodes),
                    observed_edges=edges,
                    frontier_count=frontier_count,
                    visited_count=len(visited),
                    unknown_r1=unknown_count(grid, observed, frontier, 1),
                    unknown_r2=unknown_count(grid, observed, frontier, 2),
                    free_degree=free_degree(grid, observed, frontier),
                    visit_penalty=local_visit_penalty(grid, visits, frontier, 1),
                )
                row = {
                    "map_name": Path(map_path).stem,
                    "state_step": step,
                    "candidate_rank": rank,
                    "robot_z": robot.z,
                    "robot_r": robot.r,
                    "robot_c": robot.c,
                    "frontier_z": frontier.z,
                    "frontier_r": frontier.r,
                    "frontier_c": frontier.c,
                    "first_action": action,
                    "target_score": score,
                    "is_best": 1 if score == best_score else 0,
                }
                for name, value in zip(FRONTIER_FEATURE_NAMES, feature):
                    row[name] = value
                rows_out.append(row)

        visits[robot.z][robot.r][robot.c] += 1
        visited.add(robot)
        prev_action = action_from_to(robot, path[step + 1])
    return rows_out


def write_dataset(map_dir, out_csv, observation_radius):
    map_files = sorted(Path(map_dir).glob("*.vxl"))
    all_rows = []
    for path in map_files:
        all_rows.extend(build_rows_for_map(path, observation_radius))
    if not all_rows:
        raise RuntimeError(f"no frontier rows generated from {map_dir}")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved {out_csv}, rows={len(all_rows)}, maps={len(map_files)}")


def main():
    parser = argparse.ArgumentParser(description="Build candidate-frontier rows for the SSM frontier scorer.")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--observation-radius", type=int, default=1)
    args = parser.parse_args()
    write_dataset(args.map_dir, args.out, args.observation_radius)


if __name__ == "__main__":
    main()
