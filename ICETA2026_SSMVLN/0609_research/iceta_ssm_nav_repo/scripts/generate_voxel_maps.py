#!/usr/bin/env python3
import argparse
import random
from collections import deque
from pathlib import Path


ACTION_DELTAS = [
    (0, -1, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 0, -1),
    (1, 0, 0),
    (-1, 0, 0),
]


def neighbors(shape, p):
    depth, rows, cols = shape
    z, r, c = p
    for dz, dr, dc in ACTION_DELTAS:
        q = (z + dz, r + dr, c + dc)
        if 0 <= q[0] < depth and 0 <= q[1] < rows and 0 <= q[2] < cols:
            yield q


def reachable(grid, start, target):
    shape = (len(grid), len(grid[0]), len(grid[0][0]))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == target:
            return True
        for nxt in neighbors(shape, cur):
            z, r, c = nxt
            if grid[z][r][c] != 1 and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def carve_corridor(grid, start, target):
    z, r, c = start
    tz, tr, tc = target
    grid[z][r][c] = 0
    while c != tc:
        c += 1 if tc > c else -1
        grid[z][r][c] = 0
    while r != tr:
        r += 1 if tr > r else -1
        grid[z][r][c] = 0
    while z != tz:
        z += 1 if tz > z else -1
        grid[z][r][c] = 0


def add_irregular_blocks(grid, rng, block_count):
    depth, rows, cols = len(grid), len(grid[0]), len(grid[0][0])
    for _ in range(block_count):
        z0 = rng.randrange(depth)
        r0 = rng.randrange(rows)
        c0 = rng.randrange(cols)
        dz = rng.randint(1, max(1, depth // 2))
        dr = rng.randint(1, 3)
        dc = rng.randint(1, 3)
        for z in range(z0, min(depth, z0 + dz)):
            for r in range(r0, min(rows, r0 + dr)):
                for c in range(c0, min(cols, c0 + dc)):
                    if rng.random() < 0.75:
                        grid[z][r][c] = 1


def make_map(depth, rows, cols, obstacle_prob, block_count, rng):
    start = (0, rows - 1, 0)
    for _ in range(2000):
        grid = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
        for z in range(depth):
            for r in range(rows):
                for c in range(cols):
                    if (z, r, c) != start and rng.random() < obstacle_prob:
                        grid[z][r][c] = 1

        add_irregular_blocks(grid, rng, block_count)
        grid[start[0]][start[1]][start[2]] = 0

        free = [
            (z, r, c)
            for z in range(depth)
            for r in range(rows)
            for c in range(cols)
            if grid[z][r][c] == 0 and (z, r, c) != start
        ]
        if not free:
            continue

        far = [
            p for p in free
            if abs(p[0] - start[0]) + abs(p[1] - start[1]) + abs(p[2] - start[2])
            >= (depth + rows + cols) // 3
        ] or free
        target = rng.choice(far)
        carve_corridor(grid, start, target)
        grid[target[0]][target[1]][target[2]] = 2
        if reachable(grid, start, target):
            return grid
    raise RuntimeError("failed to generate a reachable voxel map")


def save_voxel_grid(path, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for z, layer in enumerate(grid):
            if z > 0:
                f.write("\n")
            f.write(f"# z={z}\n")
            for row in layer:
                f.write(",".join(str(v) for v in row) + "\n")


def generate_split(out_root, split, count, args, seed_offset):
    rng = random.Random(args.seed + seed_offset)
    out_dir = Path(out_root) / f"voxel_{split}"
    for i in range(1, count + 1):
        grid = make_map(args.depth, args.rows, args.cols, args.obstacle_prob, args.blocks, rng)
        save_voxel_grid(out_dir / f"{split}_{i:03d}.vxl", grid)
    print(f"generated {count} {split} maps -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="maps")
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--cols", type=int, default=12)
    p.add_argument("--train", type=int, default=160)
    p.add_argument("--test", type=int, default=32)
    p.add_argument("--unseen", type=int, default=20)
    p.add_argument("--obstacle-prob", type=float, default=0.18)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--seed", type=int, default=310)
    args = p.parse_args()

    generate_split(args.out_root, "train", args.train, args, 0)
    generate_split(args.out_root, "test", args.test, args, 1000)
    generate_split(args.out_root, "unseen", args.unseen, args, 2000)


if __name__ == "__main__":
    main()
