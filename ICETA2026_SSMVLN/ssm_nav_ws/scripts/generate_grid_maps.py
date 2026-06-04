#!/usr/bin/env python3
import random
from collections import deque
from pathlib import Path

ROWS = 15
COLS = 15
START = (ROWS - 1, 0)

def neighbors(p):
    r, c = p
    for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield nr, nc

def reachable(grid, start, target):
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == target:
            return True
        for nr, nc in neighbors(cur):
            if grid[nr][nc] != 1 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False

def make_map(obstacle_prob):
    while True:
        grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) == START:
                    continue
                if random.random() < obstacle_prob:
                    grid[r][c] = 1

        free = [(r, c) for r in range(ROWS) for c in range(COLS)
                if grid[r][c] == 0 and (r, c) != START]

        if not free:
            continue

        target = random.choice(free)
        tr, tc = target
        grid[tr][tc] = 2

        if reachable(grid, START, target):
            return grid

def save_grid(path, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in grid:
            f.write(",".join(str(x) for x in row) + "\n")

def generate(split, count, obstacle_prob, seed):
    random.seed(seed)
    out_dir = Path(f"maps/grid_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        grid = make_map(obstacle_prob)
        save_grid(out_dir / f"{split}_{i:03d}.csv", grid)

    print(f"generated {count} maps -> {out_dir}")

def main():
    generate("train", 200, 0.22, 10)
    generate("test", 40, 0.22, 20)
    generate("unseen", 20, 0.25, 30)

if __name__ == "__main__":
    main()
