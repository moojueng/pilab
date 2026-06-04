#!/usr/bin/env python3
import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from voxel_nav_common import ACTIONS_3D, VoxelPos, in_bounds, load_voxel_grid, shape3, target_value_for_name
from voxel_frontier_features import FRONTIER_FEATURE_NAMES, build_frontier_feature

try:
    import torch
    from train_voxel_policy import VoxelPolicyNet
    from train_voxel_frontier_ssm import VoxelFrontierSsmNet
except Exception:
    torch = None
    VoxelPolicyNet = None
    VoxelFrontierSsmNet = None


CAMERA_DIRECTIONS = [
    (0, -1, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 0, -1),
    (1, 0, 0),
    (-1, 0, 0),
]


@dataclass
class Metrics:
    mode: str
    target_name: str = "red_chair"
    target_value: int = 2
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
    visible_voxel_hits: int = 0
    depth_rays: int = 0


class RgbdVoxelExplorer:
    def __init__(
        self,
        grid,
        mode="utility",
        max_steps=260,
        depth_range=5,
        aperture=2,
        model=None,
        target_name="red_chair",
        hybrid_ssm_weight=0.55,
    ):
        self.grid = grid
        self.depth, self.rows, self.cols = shape3(grid)
        self.mode = mode
        self.target_name = target_name
        self.target_value = target_value_for_name(target_name)
        self.max_steps = max_steps
        self.depth_range = depth_range
        self.aperture = aperture
        self.model = model
        self.hybrid_ssm_weight = max(0.0, min(1.0, float(hybrid_ssm_weight)))
        self.robot = VoxelPos(0, self.rows - 1, 0)
        self.observed = [[[-1 for _ in range(self.cols)] for _ in range(self.rows)] for _ in range(self.depth)]
        self.visits = [[[0 for _ in range(self.cols)] for _ in range(self.rows)] for _ in range(self.depth)]
        self.traj = []
        self.visited = set()
        self.node_set = set()
        self.observed_edges = 0
        self.frontier_switches = 0
        self.current_subgoal = None
        self.visible_voxel_hits = 0
        self.depth_rays = 0
        self.fallback_count = 0

    def hidden_obstacle(self, p):
        return (not in_bounds(self.grid, p)) or self.grid[p.z][p.r][p.c] == 1

    def free_observed(self, p):
        return in_bounds(self.grid, p) and self.observed[p.z][p.r][p.c] != 1 and self.observed[p.z][p.r][p.c] != -1

    def neighbors6_all(self, p):
        for dz, dr, dc in ACTIONS_3D[:6]:
            yield VoxelPos(p.z + dz, p.r + dr, p.c + dc)

    def mark_visible(self, p):
        if not in_bounds(self.grid, p):
            return False
        self.observed[p.z][p.r][p.c] = self.grid[p.z][p.r][p.c]
        self.visible_voxel_hits += 1
        return self.grid[p.z][p.r][p.c] >= 1

    def cast_ray(self, dz, dr, dc):
        self.depth_rays += 1
        length = math.sqrt(dz * dz + dr * dr + dc * dc)
        if length < 1e-6:
            return
        uz, ur, uc = dz / length, dr / length, dc / length
        last = None
        for i in range(1, self.depth_range * 4 + 1):
            t = i / 4.0
            p = VoxelPos(
                int(round(self.robot.z + uz * t)),
                int(round(self.robot.r + ur * t)),
                int(round(self.robot.c + uc * t)),
            )
            if p == last:
                continue
            last = p
            if not in_bounds(self.grid, p):
                break
            hit = self.mark_visible(p)
            if hit:
                break

    def observe_rgbd_sweep(self):
        self.mark_visible(self.robot)
        # A six-view RGB-D sweep approximates an active camera or RGB-D SLAM keyframe.
        # The agent receives only the ray-visible voxels, not a privileged local cube.
        for dz, dr, dc in CAMERA_DIRECTIONS:
            if dz != 0:
                axes = [(0, 1, 0), (0, 0, 1)]
            elif dr != 0:
                axes = [(1, 0, 0), (0, 0, 1)]
            else:
                axes = [(1, 0, 0), (0, 1, 0)]
            for a in range(-self.aperture, self.aperture + 1):
                for b in range(-self.aperture, self.aperture + 1):
                    ray = (
                        dz * 3 + axes[0][0] * a + axes[1][0] * b,
                        dr * 3 + axes[0][1] * a + axes[1][1] * b,
                        dc * 3 + axes[0][2] * a + axes[1][2] * b,
                    )
                    self.cast_ray(*ray)

    def target_visible(self):
        for z in range(self.depth):
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.observed[z][r][c] == self.target_value:
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
        return self.free_observed(p) and any(
            in_bounds(self.grid, n) and self.observed[n.z][n.r][n.c] == -1
            for n in self.neighbors6_all(p)
        )

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

    def bfs_all(self):
        q = deque([self.robot])
        parent = {self.robot: None}
        dist = {self.robot: 0}
        while q:
            cur = q.popleft()
            for n in self.neighbors6_all(cur):
                if self.free_observed(n) and n not in parent:
                    parent[n] = cur
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return parent, dist

    def action_from_to(self, a, b):
        delta = (b.z - a.z, b.r - a.r, b.c - a.c)
        for idx, d in enumerate(ACTIONS_3D[:6]):
            if d == delta:
                return idx
        return 6

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
        return 6

    def collect_frontiers(self):
        return [p for p in self.node_set if p != self.robot and self.is_frontier(p)]

    def frontier_neighbor_count(self):
        count = 0
        for p in self.node_set:
            for n in self.neighbors6_all(p):
                if in_bounds(self.grid, n) and self.observed[n.z][n.r][n.c] == -1:
                    count += 1
        return count

    def local_patch(self, radius=1):
        vals = []
        for dz in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    p = VoxelPos(self.robot.z + dz, self.robot.r + dr, self.robot.c + dc)
                    vals.append(float(self.observed[p.z][p.r][p.c]) if in_bounds(self.grid, p) else -1.0)
        return vals

    def build_policy_feature(self, prev_action):
        x = self.local_patch()
        x += [
            float(self.robot.z),
            float(self.robot.r),
            float(self.robot.c),
            float(prev_action),
            float(len(self.node_set)),
            float(self.observed_edges),
            float(self.frontier_neighbor_count()),
            float(len(self.visited)),
        ]
        return torch.tensor([x], dtype=torch.float32)

    def policy_action(self, prev_action):
        if self.model is None or torch is None:
            return None
        if getattr(self.model, "policy_kind", "action_policy") != "action_policy":
            return None
        with torch.no_grad():
            logits, rewards = self.model(self.build_policy_feature(prev_action))
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

    def unknown_count(self, p, radius):
        count = 0
        for dz in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
                    if in_bounds(self.grid, q) and self.observed[q.z][q.r][q.c] == -1:
                        count += 1
        return count

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

    def utility_score(self, p, dist):
        travel = float(dist.get(p, 10 ** 6))
        if travel >= 10 ** 6:
            return -1e9
        info = float(self.unknown_count(p, 2))
        revisit = self.local_visit_penalty(p, 1)
        dead = 1.0 if self.free_degree(p) <= 1 else 0.0
        vertical = 0.2 if p.z != self.robot.z else 0.0
        return 0.95 * info - 1.35 * travel - 2.4 * revisit - 1.3 * dead - vertical

    @staticmethod
    def normalize_scores(values):
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-8:
            return [0.5 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]

    def utility_action(self):
        frontiers = self.collect_frontiers()
        if not frontiers:
            return self.nearest_frontier_action()
        _, dist = self.bfs_all()
        best = max(frontiers, key=lambda p: self.utility_score(p, dist))
        if best != self.current_subgoal:
            self.frontier_switches += 1
        self.current_subgoal = best
        return self.first_action_to(best)

    def ssm_frontier_action(self, prev_action):
        if (
            self.model is None
            or torch is None
            or getattr(self.model, "policy_kind", "action_policy") != "frontier_scorer"
        ):
            return None
        frontiers = self.collect_frontiers()
        if not frontiers:
            return None
        _, dist = self.bfs_all()
        reachable = [p for p in frontiers if p in dist]
        if not reachable:
            return None

        features = []
        for p in reachable:
            first_action = self.first_action_to(p)
            features.append(build_frontier_feature(
                robot=self.robot,
                frontier=p,
                shape=(self.depth, self.rows, self.cols),
                graph_dist=dist[p],
                first_action=first_action,
                prev_action=prev_action,
                observed_nodes=len(self.node_set),
                observed_edges=self.observed_edges,
                frontier_count=len(frontiers),
                visited_count=len(self.visited),
                unknown_r1=self.unknown_count(p, 1),
                unknown_r2=self.unknown_count(p, 2),
                free_degree=self.free_degree(p),
                visit_penalty=self.local_visit_penalty(p, 1),
            ))
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            ssm_scores = self.model(x).squeeze(1).tolist()
        if self.mode == "ssm_utility":
            utility_scores = [self.utility_score(p, dist) for p in reachable]
            ssm_norm = self.normalize_scores([float(score) for score in ssm_scores])
            utility_norm = self.normalize_scores(utility_scores)
            w = self.hybrid_ssm_weight
            scores = [
                w * ssm_score + (1.0 - w) * utility_score
                for ssm_score, utility_score in zip(ssm_norm, utility_norm)
            ]
            best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        else:
            best_idx = max(range(len(ssm_scores)), key=lambda idx: float(ssm_scores[idx]))
        best = reachable[best_idx]
        if best != self.current_subgoal:
            self.frontier_switches += 1
        self.current_subgoal = best
        return self.first_action_to(best)

    def choose_action(self, prev_action=6):
        if self.mode == "nearest":
            return self.nearest_frontier_action()
        if self.mode in ("ssm", "ssm_utility"):
            frontier_action = self.ssm_frontier_action(prev_action)
            if frontier_action is not None and self.action_valid(frontier_action):
                return frontier_action
            action = self.policy_action(prev_action)
            if action is not None and self.action_valid(action):
                nxt = self.action_next(action)
                if self.visits[nxt.z][nxt.r][nxt.c] == 0:
                    return action
            self.fallback_count += 1
            if self.mode == "ssm":
                return self.nearest_frontier_action()
            return self.utility_action()
        return self.utility_action()

    def run(self):
        metrics = Metrics(mode=self.mode, target_name=self.target_name, target_value=self.target_value)
        if self.hidden_obstacle(self.robot):
            metrics.collisions = 1
            return metrics
        prev_action = 6
        for step in range(self.max_steps):
            self.observe_rgbd_sweep()
            self.build_graph()
            self.traj.append(self.robot)
            self.visits[self.robot.z][self.robot.r][self.robot.c] += 1
            if self.target_visible():
                metrics.success = 1
                metrics.steps = step
                metrics.target_seen_step = step
                break
            action = self.choose_action(prev_action)
            if action == 6:
                metrics.steps = step
                break
            dz, dr, dc = ACTIONS_3D[action]
            nxt = VoxelPos(self.robot.z + dz, self.robot.r + dr, self.robot.c + dc)
            if self.hidden_obstacle(nxt):
                metrics.collisions += 1
                metrics.steps = step
                break
            self.robot = nxt
            if self.robot in self.visited:
                metrics.revisits += 1
            self.visited.add(self.robot)
            prev_action = action
        else:
            metrics.steps = self.max_steps
        metrics.unique_visited = len(self.visited)
        metrics.observed_nodes = len(self.node_set)
        metrics.observed_edges = self.observed_edges
        metrics.frontier_switches = self.frontier_switches
        metrics.fallback_count = self.fallback_count
        metrics.revisit_ratio = metrics.revisits / metrics.steps if metrics.steps > 0 else 0.0
        metrics.visible_voxel_hits = self.visible_voxel_hits
        metrics.depth_rays = self.depth_rays
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


def load_ssm_policy(path):
    if torch is None or VoxelPolicyNet is None:
        raise RuntimeError("PyTorch is required for ssm/ssm_utility mode, but it could not be imported.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_type = ckpt.get("checkpoint_type", "action_policy")
    if checkpoint_type == "frontier_scorer":
        if VoxelFrontierSsmNet is None:
            raise RuntimeError("frontier_scorer checkpoint found, but VoxelFrontierSsmNet could not be imported.")
        feature_names = ckpt.get("feature_names", [])
        if feature_names and feature_names != FRONTIER_FEATURE_NAMES:
            raise RuntimeError("frontier_scorer feature order does not match this evaluator.")
        model = VoxelFrontierSsmNet(
            ckpt["input_dim"],
            ckpt.get("hidden_dim", 96),
            ckpt.get("layers", 3),
        )
        model.policy_kind = "frontier_scorer"
    else:
        model = VoxelPolicyNet(
            ckpt["input_dim"],
            ckpt.get("hidden_dim", 128),
            ckpt.get("layers", 3),
            ckpt.get("actions", 7),
        )
        model.policy_kind = "action_policy"
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", default=["nearest", "utility"], choices=["nearest", "utility", "ssm", "ssm_utility"])
    p.add_argument("--model", default="models/voxel_ssm_policy.pt", help="Voxel SSM policy checkpoint for ssm/ssm_utility modes")
    p.add_argument("--out", default="results/voxel_sim/rgbd_frontier")
    p.add_argument("--max-steps", type=int, default=260)
    p.add_argument("--depth-range", type=int, default=5)
    p.add_argument("--aperture", type=int, default=2)
    p.add_argument("--target-name", default="red_chair")
    p.add_argument("--hybrid-ssm-weight", type=float, default=0.55)
    args = p.parse_args()

    rows = []
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    model = None
    if any(mode in ("ssm", "ssm_utility") for mode in args.modes):
        model = load_ssm_policy(args.model)
    for map_path in args.maps:
        grid = load_voxel_grid(map_path)
        name = Path(map_path).stem
        for mode in args.modes:
            explorer = RgbdVoxelExplorer(
                grid,
                mode,
                args.max_steps,
                args.depth_range,
                args.aperture,
                model=model,
                target_name=args.target_name,
                hybrid_ssm_weight=args.hybrid_ssm_weight,
            )
            metrics = explorer.run()
            case_out = root / mode / name
            explorer.save_outputs(case_out)
            row = {"map": name, **metrics.__dict__}
            rows.append(row)
            with open(case_out / "metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.__dict__.keys()))
                writer.writeheader()
                writer.writerow(metrics.__dict__)
            print(row)

    fields = list(rows[0].keys()) if rows else []
    with open(root / "summary.csv", "w", newline="") as f:
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
            "avg_observed_nodes": sum(float(r["observed_nodes"]) for r in subset) / n,
            "avg_visible_voxel_hits": sum(float(r["visible_voxel_hits"]) for r in subset) / n,
            "avg_depth_rays": sum(float(r["depth_rays"]) for r in subset) / n,
        })
    with open(root / "aggregate.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggs[0].keys()))
        writer.writeheader()
        writer.writerows(aggs)
    print("\nAGGREGATE")
    for row in aggs:
        print(row)


if __name__ == "__main__":
    main()
