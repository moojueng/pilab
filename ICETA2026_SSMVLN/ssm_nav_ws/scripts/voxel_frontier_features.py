#!/usr/bin/env python3


FRONTIER_FEATURE_NAMES = [
    "robot_z_norm",
    "robot_r_norm",
    "robot_c_norm",
    "frontier_z_norm",
    "frontier_r_norm",
    "frontier_c_norm",
    "delta_z_norm",
    "delta_r_norm",
    "delta_c_norm",
    "graph_dist_norm",
    "manhattan_norm",
    "unknown_r1_norm",
    "unknown_r2_norm",
    "free_degree_norm",
    "visit_penalty_norm",
    "vertical_change",
    "first_action_norm",
    "prev_action_norm",
    "observed_nodes_norm",
    "observed_edges_norm",
    "frontier_count_norm",
    "visited_count_norm",
]


def _denom(value):
    return float(max(value - 1, 1))


def build_frontier_feature(
    robot,
    frontier,
    shape,
    graph_dist,
    first_action,
    prev_action,
    observed_nodes,
    observed_edges,
    frontier_count,
    visited_count,
    unknown_r1,
    unknown_r2,
    free_degree,
    visit_penalty,
):
    depth, rows, cols = shape
    max_manhattan = max(depth + rows + cols - 3, 1)
    dz = frontier.z - robot.z
    dr = frontier.r - robot.r
    dc = frontier.c - robot.c
    manhattan = abs(dz) + abs(dr) + abs(dc)
    return [
        robot.z / _denom(depth),
        robot.r / _denom(rows),
        robot.c / _denom(cols),
        frontier.z / _denom(depth),
        frontier.r / _denom(rows),
        frontier.c / _denom(cols),
        dz / max(float(depth), 1.0),
        dr / max(float(rows), 1.0),
        dc / max(float(cols), 1.0),
        min(float(graph_dist), 64.0) / 64.0,
        float(manhattan) / float(max_manhattan),
        min(float(unknown_r1), 27.0) / 27.0,
        min(float(unknown_r2), 125.0) / 125.0,
        min(float(free_degree), 6.0) / 6.0,
        min(float(visit_penalty), 30.0) / 30.0,
        1.0 if frontier.z != robot.z else 0.0,
        float(first_action) / 6.0,
        float(prev_action) / 6.0,
        min(float(observed_nodes), 2048.0) / 2048.0,
        min(float(observed_edges), 8192.0) / 8192.0,
        min(float(frontier_count), 256.0) / 256.0,
        min(float(visited_count), 1024.0) / 1024.0,
    ]

