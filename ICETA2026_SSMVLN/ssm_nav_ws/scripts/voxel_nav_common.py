#!/usr/bin/env python3
from collections import deque
from dataclasses import dataclass
import re
import zlib


ACTIONS_3D = [
    (0, -1, 0),   # north
    (0, 0, 1),    # east
    (0, 1, 0),    # south
    (0, 0, -1),   # west
    (1, 0, 0),    # up
    (-1, 0, 0),   # down
    (0, 0, 0),    # stop
]

TARGET_NAME_TO_VALUE = {
    "red_chair": 2,
    "blue_bed": 3,
    "green_table": 4,
}

TARGET_VALUE_TO_NAME = {value: name for name, value in TARGET_NAME_TO_VALUE.items()}
SEMANTIC_VALUE_MIN = 2
DYNAMIC_TARGET_VALUE_BASE = 100
DYNAMIC_TARGET_VALUE_BUCKETS = 9000


@dataclass(frozen=True, order=True)
class VoxelPos:
    z: int
    r: int
    c: int


def load_voxel_grid(path):
    layers = []
    cur = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if cur:
                    layers.append(cur)
                    cur = []
                continue
            if line.startswith("#"):
                continue
            cur.append([int(x) for x in line.split(",")])
    if cur:
        layers.append(cur)
    if not layers:
        raise ValueError(f"empty voxel map: {path}")
    return layers


def shape3(grid):
    return len(grid), len(grid[0]), len(grid[0][0])


def in_bounds(grid, p):
    d, rows, cols = shape3(grid)
    return 0 <= p.z < d and 0 <= p.r < rows and 0 <= p.c < cols


def free_like(grid, p):
    return in_bounds(grid, p) and grid[p.z][p.r][p.c] != 1


def target_value_for_name(target_name):
    name = normalize_target_name(target_name)
    if name in TARGET_NAME_TO_VALUE:
        return TARGET_NAME_TO_VALUE[name]
    crc = zlib.crc32(name.encode("utf-8")) % DYNAMIC_TARGET_VALUE_BUCKETS
    return DYNAMIC_TARGET_VALUE_BASE + crc


def normalize_target_name(target_name):
    name = str(target_name or "red_chair").strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "red_chair"


def semantic_target_values(extra_target_names=None):
    values = dict(TARGET_NAME_TO_VALUE)
    for name in extra_target_names or []:
        clean = normalize_target_name(name)
        values[clean] = target_value_for_name(clean)
    return values


def is_semantic_value(value):
    return int(value) >= SEMANTIC_VALUE_MIN


def neighbors6(grid, p):
    for a, (dz, dr, dc) in enumerate(ACTIONS_3D[:6]):
        q = VoxelPos(p.z + dz, p.r + dr, p.c + dc)
        if free_like(grid, q):
            yield a, q


def find_target(grid):
    return find_target_value(grid, 2)


def find_target_value(grid, target_value):
    d, rows, cols = shape3(grid)
    for z in range(d):
        for r in range(rows):
            for c in range(cols):
                if grid[z][r][c] == target_value:
                    return VoxelPos(z, r, c)
    raise ValueError(f"target voxel {target_value} not found")


def bfs_path(grid, start, target):
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            break
        for _, nxt in neighbors6(grid, cur):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    if target not in parent:
        return []
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def action_from_to(a, b):
    delta = (b.z - a.z, b.r - a.r, b.c - a.c)
    for idx, d in enumerate(ACTIONS_3D):
        if d == delta:
            return idx
    return 6
