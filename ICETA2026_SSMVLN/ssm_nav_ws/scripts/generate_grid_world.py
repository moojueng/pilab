#!/usr/bin/env python3
import argparse
from pathlib import Path


def load_grid(path):
    grid = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grid.append([int(x.strip()) for x in line.split(",")])
    return grid


def box_model(name, x, y, z, sx, sy, sz, color):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def cylinder_model(name, x, y, z, radius, length, color):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>{radius:.3f}</radius>
              <length>{length:.3f}</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>{radius:.3f}</radius>
              <length>{length:.3f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def generate_world(grid, cell_size, wall_height):
    rows = len(grid)
    cols = len(grid[0])
    origin_x = -(cols - 1) * cell_size / 2.0
    origin_y = (rows - 1) * cell_size / 2.0

    floor_sx = cols * cell_size + cell_size
    floor_sy = rows * cell_size + cell_size

    models = []
    models.append(box_model(
        "grid_floor", 0.0, 0.0, -0.01,
        floor_sx, floor_sy, 0.02,
        "0.65 0.65 0.65 1"
    ))

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x = origin_x + c * cell_size
            y = origin_y - r * cell_size

            if value == 1:
                models.append(box_model(
                    f"wall_{r}_{c}", x, y, wall_height / 2.0,
                    cell_size * 1.00, cell_size * 1.00, wall_height,
                    "0.05 0.05 0.05 1"
                ))
            else:
                models.append(box_model(
                    f"cell_{r}_{c}", x, y, 0.001,
                    cell_size * 0.96, cell_size * 0.96, 0.002,
                    "0.85 0.85 0.85 1"
                ))

    target_r = 0
    target_c = 0
    target_x = origin_x + target_c * cell_size
    target_y = origin_y - target_r * cell_size
    models.append(cylinder_model(
        "target_cylinder",
        target_x,
        target_y,
        0.35,
        0.28,
        0.70,
        "1.0 0.0 0.0 1"
    ))

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="ssm_2d_grid_world">
    <plugin name="gazebo_ros_init" filename="libgazebo_ros_init.so"/>
    <plugin name="gazebo_ros_factory" filename="libgazebo_ros_factory.so"/>

    <include>
      <uri>model://sun</uri>
    </include>

    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    {''.join(models)}
  </world>
</sdf>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--wall-height", type=float, default=0.7)
    args = parser.parse_args()

    grid = load_grid(args.map)
    world = generate_world(grid, args.cell_size, args.wall_height)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(world)

    print(f"Generated Gazebo grid world: {out_path}")


if __name__ == "__main__":
    main()

