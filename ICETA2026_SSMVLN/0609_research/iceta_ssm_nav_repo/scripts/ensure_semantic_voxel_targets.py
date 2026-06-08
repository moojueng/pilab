#!/usr/bin/env python3
import argparse
import random
from collections import deque
from pathlib import Path

from voxel_nav_common import (
    ACTIONS_3D,
    VoxelPos,
    free_like,
    in_bounds,
    load_voxel_grid,
    normalize_target_name,
    semantic_target_values,
    shape3,
)
from generate_voxel_maps import save_voxel_grid


def start_for(grid):
    _, rows, _ = shape3(grid)
    return VoxelPos(0, rows - 1, 0)


def neighbors6(grid, p):
    for dz, dr, dc in ACTIONS_3D[:6]:
        q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
        if free_like(grid, q):
            yield q


def reachable_free(grid, start):
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in neighbors6(grid, cur):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def value_exists(grid, value):
    depth, rows, cols = shape3(grid)
    for z in range(depth):
        for r in range(rows):
            for c in range(cols):
                if grid[z][r][c] == value:
                    return True
    return False


def target_anchor(grid, target_name):
    depth, rows, cols = shape3(grid)
    name = normalize_target_name(target_name)
    if name == "red_chair":
        return VoxelPos(depth - 1, 0, cols - 1)
    if name == "blue_bed":
        return VoxelPos(depth - 1, rows // 4, cols - 1)
    if name == "green_table":
        return VoxelPos(depth // 2, rows // 2, cols // 2)
    if name == "yellow_sofa":
        return VoxelPos(depth // 2, 0, cols // 3)
    if name == "black_chair":
        return VoxelPos(depth - 1, rows // 3, cols // 2)
    if name == "black_gate":
        return VoxelPos(0, 0, cols - 1)
    return VoxelPos(depth - 1, 0, cols - 1)


def place_target(grid, value, rng, occupied_values, target_name=None, placement="random"):
    start = start_for(grid)
    reachable = reachable_free(grid, start)
    candidates = [
        p for p in reachable
        if p != start and grid[p.z][p.r][p.c] == 0
    ]
    if not candidates:
        candidates = [
            p for p in reachable
            if p != start and grid[p.z][p.r][p.c] not in occupied_values and grid[p.z][p.r][p.c] != 1
        ]
    if not candidates:
        raise RuntimeError("no reachable free voxel for semantic target placement")
    max_dist = max(abs(p.z - start.z) + abs(p.r - start.r) + abs(p.c - start.c) for p in candidates)
    far = [
        p for p in candidates
        if abs(p.z - start.z) + abs(p.r - start.r) + abs(p.c - start.c) >= max(1, int(max_dist * 0.6))
    ] or candidates
    if placement == "semantic_prior" and target_name:
        anchor = target_anchor(grid, target_name)
        denom = max(sum(shape3(grid)), 1)
        def score(p):
            dist_anchor = abs(p.z - anchor.z) + abs(p.r - anchor.r) + abs(p.c - anchor.c)
            dist_start = abs(p.z - start.z) + abs(p.r - start.r) + abs(p.c - start.c)
            return -float(dist_anchor) / denom + 0.20 * float(dist_start) / denom + 0.01 * rng.random()
        p = max(candidates, key=score)
    else:
        p = rng.choice(far)
    grid[p.z][p.r][p.c] = value


def clear_existing_targets(grid, values):
    depth, rows, cols = shape3(grid)
    value_set = set(values)
    for z in range(depth):
        for r in range(rows):
            for c in range(cols):
                if grid[z][r][c] in value_set:
                    grid[z][r][c] = 0


def ensure_map(path, seed, target_names=None, placement="random", replace_existing=False):
    grid = load_voxel_grid(path)
    rng = random.Random(seed + sum(ord(ch) for ch in Path(path).stem))
    targets = semantic_target_values(target_names)
    occupied_values = set(targets.values())
    if replace_existing:
        clear_existing_targets(grid, occupied_values)
    for name, value in sorted(targets.items(), key=lambda item: item[1]):
        if replace_existing or not value_exists(grid, value):
            place_target(grid, value, rng, occupied_values, target_name=name, placement=placement)
    save_voxel_grid(Path(path), grid)
    print(f"semantic targets ensured: {path}")


def ensure_dir(map_dir, seed=913, target_names=None, placement="random", replace_existing=False):
    for path in sorted(Path(map_dir).glob("*.vxl")):
        ensure_map(path, seed, target_names=target_names, placement=placement, replace_existing=replace_existing)


def main():
    parser = argparse.ArgumentParser(description="Ensure voxel maps contain fixed and command-conditioned semantic targets.")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--placement", default="random", choices=["random", "semantic_prior"])
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--target-name",
        action="append",
        default=[],
        help="Additional command-conditioned target class to place, e.g. yellow_sofa. Can be repeated.",
    )
    args = parser.parse_args()
    ensure_dir(
        args.map_dir,
        seed=args.seed,
        target_names=args.target_name,
        placement=args.placement,
        replace_existing=args.replace_existing,
    )


if __name__ == "__main__":
    main()
