#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from train_voxel_policy import VoxelPolicyNet
from voxel_nav_common import ACTIONS_3D, VoxelPos, in_bounds, load_voxel_grid, shape3


@dataclass
class Metrics:
    mode: str
    success: int = 0
    steps: int = 0
    collisions: int = 0
    revisits: int = 0
    revisit_ratio: float = 0.0
    unique_visited: int = 0
    observed_nodes: int = 0
    observed_edges: int = 0
    target_seen_step: int = -1
    fallback_count: int = 0
    frontier_switches: int = 0
    vertical_moves: int = 0


class VoxelEvaluator:
    def __init__(self, grid, model, mode="utility", vision_radius=1, max_steps=300):
        self.grid = grid
        self.depth, self.rows, self.cols = shape3(grid)
        self.observed = [[[-1 for _ in range(self.cols)] for _ in range(self.rows)] for _ in range(self.depth)]
        self.visits = [[[0 for _ in range(self.cols)] for _ in range(self.rows)] for _ in range(self.depth)]
        self.robot = VoxelPos(0, self.rows - 1, 0)
        self.model = model
        self.mode = mode
        self.vision_radius = vision_radius
        self.max_steps = max_steps
        self.traj = []
        self.visited = set()
        self.node_set = set()
        self.observed_edges = 0
        self.fallback_count = 0
        self.frontier_switches = 0
        self.current_subgoal = None
        self.commit_left = 0

    def hidden_obstacle(self, p):
        return (not in_bounds(self.grid, p)) or self.grid[p.z][p.r][p.c] == 1

    def free_observed(self, p):
        return in_bounds(self.grid, p) and self.observed[p.z][p.r][p.c] in (0, 2)

    def neighbors6_all(self, p):
        for dz, dr, dc in ACTIONS_3D[:6]:
            yield VoxelPos(p.z + dz, p.r + dr, p.c + dc)

    def observe(self):
        r = self.vision_radius
        for dz in range(-r, r + 1):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    p = VoxelPos(self.robot.z + dz, self.robot.r + dr, self.robot.c + dc)
                    if in_bounds(self.grid, p):
                        self.observed[p.z][p.r][p.c] = self.grid[p.z][p.r][p.c]

    def target_visible(self):
        r = self.vision_radius
        for dz in range(-r, r + 1):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    p = VoxelPos(self.robot.z + dz, self.robot.r + dr, self.robot.c + dc)
                    if in_bounds(self.grid, p) and self.observed[p.z][p.r][p.c] == 2:
                        return True
        return False

    def build_graph(self):
        self.node_set = set()
        self.observed_edges = 0
        for z in range(self.depth):
            for r in range(self.rows):
                for c in range(self.cols):
                    p = VoxelPos(z, r, c)
                    if self.free_observed(p):
                        self.node_set.add(p)
        for p in self.node_set:
            for n in self.neighbors6_all(p):
                if n in self.node_set:
                    self.observed_edges += 1

    def is_frontier(self, p):
        return self.free_observed(p) and any(in_bounds(self.grid, n) and self.observed[n.z][n.r][n.c] == -1 for n in self.neighbors6_all(p))

    def build_feature(self, prev_action):
        x = []
        for dz in range(-1, 2):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    p = VoxelPos(self.robot.z + dz, self.robot.r + dr, self.robot.c + dc)
                    x.append(float(self.observed[p.z][p.r][p.c]) if in_bounds(self.grid, p) else -1.0)
        frontier_count = 0
        for p in self.node_set:
            for n in self.neighbors6_all(p):
                if in_bounds(self.grid, n) and self.observed[n.z][n.r][n.c] == -1:
                    frontier_count += 1
        x += [
            float(self.robot.z), float(self.robot.r), float(self.robot.c),
            float(prev_action), float(len(self.node_set)), float(self.observed_edges),
            float(frontier_count), float(len(self.visited)),
        ]
        return torch.tensor([x], dtype=torch.float32)

    def policy_action(self, prev_action):
        with torch.no_grad():
            logits, rewards = self.model(self.build_feature(prev_action))
            scores = rewards[0] + 0.2 * logits[0]
            return int(torch.argmax(scores).item())

    def action_next(self, action):
        dz, dr, dc = ACTIONS_3D[action]
        return VoxelPos(self.robot.z + dz, self.robot.r + dr, self.robot.c + dc)

    def action_valid(self, action):
        if action not in range(6):
            return False
        nxt = self.action_next(action)
        return self.free_observed(nxt) and not self.hidden_obstacle(nxt)

    def action_from_to(self, a, b):
        delta = (b.z - a.z, b.r - a.r, b.c - a.c)
        for idx, d in enumerate(ACTIONS_3D):
            if d == delta:
                return idx
        return 6

    def bfs_parent(self, goal=None):
        q = deque([self.robot])
        parent = {self.robot: None}
        dist = {self.robot: 0}
        found = None
        while q:
            cur = q.popleft()
            if goal is not None and cur == goal:
                found = cur
                break
            if goal is None and cur != self.robot and self.is_frontier(cur):
                found = cur
                break
            for n in self.neighbors6_all(cur):
                if self.free_observed(n) and n not in parent:
                    parent[n] = cur
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return found, parent, dist

    def first_action_to(self, goal):
        found, parent, _ = self.bfs_parent(goal)
        if found is None:
            return 6
        cur = goal
        while parent.get(cur) is not None and parent[cur] != self.robot:
            cur = parent[cur]
        return self.action_from_to(self.robot, cur)

    def nearest_frontier_action(self):
        found, parent, _ = self.bfs_parent(None)
        if found is not None:
            cur = found
            while parent.get(cur) is not None and parent[cur] != self.robot:
                cur = parent[cur]
            return self.action_from_to(self.robot, cur)
        for a in range(6):
            if self.action_valid(a):
                return a
        return 6

    def collect_frontiers(self):
        return [p for p in self.node_set if p != self.robot and self.is_frontier(p)]

    def unknown_count(self, p, radius):
        cnt = 0
        for dz in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                    if in_bounds(self.grid, q) and self.observed[q.z][q.r][q.c] == -1:
                        cnt += 1
        return cnt

    def local_visit_penalty(self, p, radius=1):
        total = 0
        for dz in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                    if in_bounds(self.grid, q):
                        total += min(self.visits[q.z][q.r][q.c], 5)
        return float(total)

    def free_degree(self, p):
        return sum(1 for n in self.neighbors6_all(p) if self.free_observed(n))

    def best_frontier(self, prev_action):
        frontiers = self.collect_frontiers()
        if not frontiers:
            return None
        _, _, dist = self.bfs_parent(None)

        def score(p):
            travel = float(dist.get(p, 10 ** 6))
            if travel >= 10 ** 6:
                return -1e9
            info = float(self.unknown_count(p, self.vision_radius + 1))
            prior = float(self.unknown_count(p, 2))
            revisit = self.local_visit_penalty(p, 1)
            dead = 1.0 if self.free_degree(p) <= 1 else 0.0
            vertical = 0.2 if p.z != self.robot.z else 0.0
            turn = 0.0 if prev_action not in range(6) else (0.0 if self.first_action_to(p) == prev_action else 1.0)
            semantic_proxy = info / 12.0
            return 1.0 * semantic_proxy + 0.85 * info + 0.35 * prior - 1.25 * travel - 2.2 * revisit - 1.1 * dead - vertical - 0.45 * turn

        return max(frontiers, key=score)

    def utility_action(self, prev_action):
        if self.mode == "utility_commit" and self.current_subgoal is not None and self.commit_left > 0 and self.free_observed(self.current_subgoal):
            action = self.first_action_to(self.current_subgoal)
            if action in range(6):
                self.commit_left -= 1
                return action
        best = self.best_frontier(prev_action)
        if best is not None:
            if best != self.current_subgoal:
                self.frontier_switches += 1
            self.current_subgoal = best
            self.commit_left = 5 if self.mode == "utility_commit" else 0
            action = self.first_action_to(best)
            if action in range(6):
                return action
        return self.nearest_frontier_action()

    def choose_action(self, policy_action, prev_action):
        if self.mode == "policy_only":
            if self.action_valid(policy_action):
                return policy_action
            self.fallback_count += 1
            return self.nearest_frontier_action()
        if self.mode == "nearest":
            if self.action_valid(policy_action) and self.visits[self.action_next(policy_action).z][self.action_next(policy_action).r][self.action_next(policy_action).c] == 0:
                return policy_action
            self.fallback_count += 1
            return self.nearest_frontier_action()
        if self.action_valid(policy_action) and self.visits[self.action_next(policy_action).z][self.action_next(policy_action).r][self.action_next(policy_action).c] == 0:
            return policy_action
        self.fallback_count += 1
        return self.utility_action(prev_action)

    def run(self):
        metrics = Metrics(mode=self.mode)
        if self.hidden_obstacle(self.robot):
            metrics.collisions = 1
            return metrics
        prev_action = 6
        for step in range(self.max_steps):
            self.observe()
            self.build_graph()
            self.traj.append(self.robot)
            self.visits[self.robot.z][self.robot.r][self.robot.c] += 1
            if self.target_visible():
                metrics.success = 1
                metrics.steps = step
                metrics.target_seen_step = step
                break
            action = self.choose_action(self.policy_action(prev_action), prev_action)
            if action == 6:
                metrics.steps = step
                break
            if action in (4, 5):
                metrics.vertical_moves += 1
            nxt = self.action_next(action)
            if self.hidden_obstacle(nxt):
                metrics.collisions += 1
                self.fallback_count += 1
                action = self.utility_action(prev_action)
                nxt = self.action_next(action)
                if self.hidden_obstacle(nxt):
                    metrics.steps = step
                    break
            self.robot = nxt
            if self.robot in self.visited:
                metrics.revisits += 1
            self.visited.add(self.robot)
            prev_action = action
        else:
            metrics.steps = self.max_steps
        metrics.observed_nodes = len(self.node_set)
        metrics.observed_edges = self.observed_edges
        metrics.unique_visited = len(self.visited)
        metrics.fallback_count = self.fallback_count
        metrics.frontier_switches = self.frontier_switches
        metrics.revisit_ratio = metrics.revisits / metrics.steps if metrics.steps > 0 else 0.0
        return metrics

    def save_outputs(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "trajectory.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "z", "r", "c"])
            writer.writerows((i, p.z, p.r, p.c) for i, p in enumerate(self.traj))
        with open(out / "observed_voxels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["z", "r", "c", "value", "visits"])
            for z in range(self.depth):
                for r in range(self.rows):
                    for c in range(self.cols):
                        writer.writerow([z, r, c, self.observed[z][r][c], self.visits[z][r][c]])
        self.save_projection(out / "path_projection.png")

    def save_projection(self, path):
        scale = 28
        img = Image.new("RGB", (self.cols * scale, self.rows * scale), "white")
        draw = ImageDraw.Draw(img)
        for r in range(self.rows):
            for c in range(self.cols):
                vals = [self.observed[z][r][c] for z in range(self.depth)]
                if 2 in vals:
                    color = (255, 40, 40)
                elif any(v == 1 for v in vals):
                    color = (20, 20, 20)
                elif any(v == 0 for v in vals):
                    color = (245, 245, 245)
                else:
                    color = (165, 165, 165)
                draw.rectangle([c * scale, r * scale, (c + 1) * scale - 1, (r + 1) * scale - 1], fill=color, outline=(120, 120, 120))
        palette = [(40, 130, 255), (20, 170, 120), (255, 190, 40), (180, 80, 220), (255, 110, 60)]
        for p in self.traj:
            color = palette[p.z % len(palette)]
            pad = 5
            draw.ellipse([p.c * scale + pad, p.r * scale + pad, (p.c + 1) * scale - pad, (p.r + 1) * scale - pad], fill=color)
        if self.traj:
            p = self.traj[0]
            draw.rectangle([p.c * scale + 6, p.r * scale + 6, (p.c + 1) * scale - 6, (p.r + 1) * scale - 6], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        p = self.robot
        draw.rectangle([p.c * scale + 4, p.r * scale + 4, (p.c + 1) * scale - 4, (p.r + 1) * scale - 4], fill=(0, 230, 80))
        img.save(path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = VoxelPolicyNet(ckpt["input_dim"], ckpt.get("hidden_dim", 128), ckpt.get("layers", 3), ckpt.get("actions", 7))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", nargs="+", required=True)
    p.add_argument("--model", default="models/voxel_ssm_policy.pt")
    p.add_argument("--modes", nargs="+", default=["nearest", "utility"])
    p.add_argument("--out", default="results/voxel_sim/frontier_modes")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--vision-radius", type=int, default=1)
    args = p.parse_args()

    model = load_model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for map_path in args.maps:
        name = Path(map_path).stem
        grid = load_voxel_grid(map_path)
        for mode in args.modes:
            ev = VoxelEvaluator(grid, model, mode, args.vision_radius, args.max_steps)
            metrics = ev.run()
            case_out = out / mode / name
            ev.save_outputs(case_out)
            row = {"map": name, **metrics.__dict__}
            rows.append(row)
            with open(case_out / "metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.__dict__.keys()))
                writer.writeheader()
                writer.writerow(metrics.__dict__)
            print(row)

    fields = list(rows[0].keys()) if rows else []
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggs = []
    for mode in args.modes:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        n = len(subset)
        successful = [r for r in subset if int(r["success"]) == 1]
        aggs.append({
            "mode": mode,
            "maps": n,
            "success_rate": sum(int(r["success"]) for r in subset) / n,
            "avg_steps_all": sum(float(r["steps"]) for r in subset) / n,
            "avg_steps_success": sum(float(r["steps"]) for r in successful) / max(len(successful), 1),
            "avg_revisit_ratio": sum(float(r["revisit_ratio"]) for r in subset) / n,
            "avg_fallback_count": sum(float(r["fallback_count"]) for r in subset) / n,
            "avg_frontier_switches": sum(float(r["frontier_switches"]) for r in subset) / n,
            "avg_vertical_moves": sum(float(r["vertical_moves"]) for r in subset) / n,
        })
    with open(out / "aggregate.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggs[0].keys()))
        writer.writeheader()
        writer.writerows(aggs)
    print("\nAGGREGATE")
    for row in aggs:
        print(row)


if __name__ == "__main__":
    main()
