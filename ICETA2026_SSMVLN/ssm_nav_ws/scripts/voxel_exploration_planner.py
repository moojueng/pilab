#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from voxel_nav_common import ACTIONS_3D, VoxelPos, find_target, in_bounds, load_voxel_grid, shape3


@dataclass
class PlannedRoute:
    goal: VoxelPos | None
    path: list[VoxelPos]
    selected_score: float = 0.0
    unknown_gain: int = 0
    reachable_nodes: int = 0
    frontier_count: int = 0


class VoxelExplorationPlanner:
    """6-neighbor 3D planner over a partial observed voxel graph."""

    def __init__(self, hidden_grid, observed=None):
        self.hidden_grid = hidden_grid
        self.depth, self.rows, self.cols = shape3(hidden_grid)
        self.observed = observed or self.full_observation(hidden_grid)

    @staticmethod
    def full_observation(grid):
        depth, rows, cols = shape3(grid)
        return [[[grid[z][r][c] for c in range(cols)] for r in range(rows)] for z in range(depth)]

    @classmethod
    def from_observed_csv(cls, hidden_grid, path):
        depth, rows, cols = shape3(hidden_grid)
        observed = [[[-1 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if {"z", "r", "c", "value"}.issubset(row):
                    z = int(row["z"])
                    r = int(row["r"])
                    c = int(row["c"])
                    value = int(row["value"])
                elif {"ix", "iy", "iz", "value"}.issubset(row):
                    z = int(row["iz"])
                    r = int(row["iy"])
                    c = int(row["ix"])
                    value = int(row["value"])
                else:
                    raise ValueError(f"unsupported observed voxel csv columns: {reader.fieldnames}")
                p = VoxelPos(z, r, c)
                if in_bounds(hidden_grid, p):
                    observed[z][r][c] = value
        return cls(hidden_grid, observed)

    def free_observed(self, p):
        return in_bounds(self.hidden_grid, p) and self.observed[p.z][p.r][p.c] in (0, 2)

    def neighbors6(self, p):
        for dz, dr, dc in ACTIONS_3D[:6]:
            q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
            if self.free_observed(q):
                yield q

    def observed_nodes(self):
        nodes = []
        for z in range(self.depth):
            for r in range(self.rows):
                for c in range(self.cols):
                    p = VoxelPos(z, r, c)
                    if self.free_observed(p):
                        nodes.append(p)
        return nodes

    def is_frontier(self, p):
        if not self.free_observed(p):
            return False
        for dz, dr, dc in ACTIONS_3D[:6]:
            q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
            if in_bounds(self.hidden_grid, q) and self.observed[q.z][q.r][q.c] == -1:
                return True
        return False

    def bfs(self, start, goal=None):
        if not self.free_observed(start):
            return {}, {}
        q = deque([start])
        parent = {start: None}
        dist = {start: 0}
        while q:
            cur = q.popleft()
            if goal is not None and cur == goal:
                break
            for nxt in self.neighbors6(cur):
                if nxt not in parent:
                    parent[nxt] = cur
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return parent, dist

    @staticmethod
    def reconstruct_path(parent, start, goal):
        if goal not in parent:
            return []
        cur = goal
        path = [cur]
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def unknown_count(self, p, radius=2):
        count = 0
        for dz in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                    if in_bounds(self.hidden_grid, q) and self.observed[q.z][q.r][q.c] == -1:
                        count += 1
        return count

    def plan_to_target(self, start):
        target = find_target(self.hidden_grid)
        parent, dist = self.bfs(start, target)
        path = self.reconstruct_path(parent, start, target)
        return PlannedRoute(
            goal=target if path else None,
            path=path,
            selected_score=float(dist.get(target, 0)),
            reachable_nodes=len(dist),
        )

    def plan_to_frontier(self, start, selector="utility"):
        parent, dist = self.bfs(start)
        reachable = set(dist)
        frontiers = [p for p in self.observed_nodes() if p in reachable and p != start and self.is_frontier(p)]
        if not frontiers:
            return PlannedRoute(goal=None, path=[], reachable_nodes=len(dist), frontier_count=0)

        def score(p):
            travel = float(dist[p])
            gain = float(self.unknown_count(p, radius=2))
            vertical = 0.25 if p.z != start.z else 0.0
            if selector == "nearest":
                return -travel
            return 1.0 * gain - 1.35 * travel - vertical

        best = max(frontiers, key=score)
        return PlannedRoute(
            goal=best,
            path=self.reconstruct_path(parent, start, best),
            selected_score=score(best),
            unknown_gain=self.unknown_count(best, radius=2),
            reachable_nodes=len(dist),
            frontier_count=len(frontiers),
        )


def action_between(a, b):
    delta = (b.z - a.z, b.r - a.r, b.c - a.c)
    for idx, step in enumerate(ACTIONS_3D[:6]):
        if delta == step:
            return idx
    return 6


def write_path(path, route):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "z", "r", "c", "action_from_prev"])
        for i, p in enumerate(route.path):
            action = 6 if i == 0 else action_between(route.path[i - 1], p)
            writer.writerow([i, p.z, p.r, p.c, action])


def main():
    parser = argparse.ArgumentParser(description="Plan a 3D voxel exploration route with 6-neighbor BFS.")
    parser.add_argument("--map", required=True, help="Hidden .vxl map used for shape and optional target planning.")
    parser.add_argument("--observed", help="Observed voxel CSV. If omitted, the full map is treated as observed.")
    parser.add_argument("--start", nargs=3, type=int, metavar=("Z", "R", "C"), default=None)
    parser.add_argument("--mode", choices=["frontier", "target"], default="frontier")
    parser.add_argument("--selector", choices=["utility", "nearest"], default="utility")
    parser.add_argument("--out", default="results/voxel_sim/planned_3d_path.csv")
    args = parser.parse_args()

    grid = load_voxel_grid(args.map)
    _, rows, _ = shape3(grid)
    start = VoxelPos(*args.start) if args.start else VoxelPos(0, rows - 1, 0)
    planner = (
        VoxelExplorationPlanner.from_observed_csv(grid, args.observed)
        if args.observed
        else VoxelExplorationPlanner(grid)
    )

    route = (
        planner.plan_to_target(start)
        if args.mode == "target"
        else planner.plan_to_frontier(start, selector=args.selector)
    )
    write_path(args.out, route)

    goal_text = "none" if route.goal is None else f"({route.goal.z},{route.goal.r},{route.goal.c})"
    print(
        "planned",
        f"mode={args.mode}",
        f"selector={args.selector}",
        f"start=({start.z},{start.r},{start.c})",
        f"goal={goal_text}",
        f"path_len={len(route.path)}",
        f"frontiers={route.frontier_count}",
        f"reachable={route.reachable_nodes}",
        f"unknown_gain={route.unknown_gain}",
        f"score={route.selected_score:.3f}",
        f"out={args.out}",
    )


if __name__ == "__main__":
    main()
