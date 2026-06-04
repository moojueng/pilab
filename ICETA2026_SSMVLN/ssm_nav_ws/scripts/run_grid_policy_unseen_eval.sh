#!/usr/bin/env bash
set -e

cd /home/mj/my_research/ssm_nav_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

mkdir -p results/grid_sim/policy_unseen

echo "map,success,steps,collisions,revisits,observed_nodes,observed_edges,target_seen_step,fallback_count" \
  > results/grid_sim/policy_unseen/summary.csv

for map_file in maps/grid_unseen/unseen_*.csv; do
  name=$(basename "$map_file" .csv)
  out_dir="results/grid_sim/policy_unseen/${name}"
  mkdir -p "$out_dir"

  echo "running ${name}"

  ros2 run s_nav_core grid_policy_eval "$map_file" models/grid_ssm_policy.onnx

  cp results/grid_sim/policy_metrics.csv "$out_dir/metrics.csv"
  cp results/grid_sim/policy_trajectory.csv "$out_dir/trajectory.csv"
  cp results/grid_sim/policy_observed_map.csv "$out_dir/observed_map.csv"
  cp results/grid_sim/policy_path.ppm "$out_dir/path.ppm"

  python3 - <<PY
from PIL import Image
from pathlib import Path
Image.open(Path("${out_dir}/path.ppm")).save(Path("${out_dir}/path.png"))
PY

  tail -n 1 "$out_dir/metrics.csv" | sed "s/^/${name},/" \
    >> results/grid_sim/policy_unseen/summary.csv
done

echo "==== summary ===="
cat results/grid_sim/policy_unseen/summary.csv
