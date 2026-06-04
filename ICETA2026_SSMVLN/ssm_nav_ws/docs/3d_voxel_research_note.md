# 3D Voxel Navigation Development Note

## 2026-05-26 Update: Local LLM/VLN + One-Time SSM Frontier Scorer

The current research positioning is now:

- core exploration: online 3D voxel-frontier graph from RGB-D observations
- language/VLN role: natural-language command grounding into a target spec, not global path planning
- SSM role: one-time initial training of a lightweight frontier scorer, then inference only in new maps
- path generation: BFS path reconstruction through the observed free-space graph toward the selected frontier

Important wording:

- This is not "SSM with no learning." The SSM frontier scorer is trained once on synthetic voxel-frontier candidate data.
- This is "no per-environment fine-tuning." New unseen maps use only online RGB-D observations, observed graph updates, frontier candidates, and the frozen SSM scorer.
- The LLM/VLN module is used for command grounding: e.g. "전체 집을 순찰하면서 빨간 의자를 발견하면 로그 남겨" -> `target=red_chair`, `action=log`, `mission_mode=coverage_patrol`.
- Exploration itself does not depend on cloud LLM calls. `scripts/local_llm_grounding_server.py` serves a real local HuggingFace instruct model. Current tested model: `Qwen/Qwen2.5-1.5B-Instruct`.
- For professor-facing runs, `mission.json` must show `llm_used=true` and `grounding_type=local_hf_llm:Qwen/Qwen2.5-1.5B-Instruct`. `llm_used=false` is only a fallback/debug run and should not be presented as the LLM result.

New files:

- `scripts/local_vln_llm.py`: local LLM/VLN command grounding with symbolic fallback
- `scripts/local_llm_grounding_server.py`: real local HuggingFace LLM server for command grounding
- `scripts/build_voxel_frontier_dataset.py`: candidate-frontier dataset builder
- `scripts/train_voxel_frontier_ssm.py`: one-time SSM-style frontier scorer training
- `scripts/voxel_frontier_features.py`: shared frontier feature order
- `scripts/run_vln_ssm_voxel_study.py`: end-to-end map generation, training, evaluation, and viewer creation
- `scripts/build_voxel_study_dashboard.py`: static dashboard for multi-map 3D voxel study inspection

Verified run:

```bash
python3 scripts/run_vln_ssm_voxel_study.py \
  --regenerate-maps \
  --rebuild-dataset \
  --retrain \
  --epochs 35 \
  --cpu \
  --llm-provider symbolic
```

Output:

- model: `models/voxel_frontier_ssm.pt`
- train log: `results/vln_ssm_voxel_study/train_frontier_ssm_log.csv`
- aggregate: `results/vln_ssm_voxel_study/aggregate.csv`
- viewer: `results/vln_ssm_voxel_study/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html`
- dashboard: `results/vln_ssm_voxel_study/dashboard.html`

Result on 4 random unseen voxel maps:

| mode | success rate | avg steps success | avg revisit ratio |
| --- | ---: | ---: | ---: |
| utility | 1.0 | 17.25 | 0.0435 |
| ssm_utility | 1.0 | 16.25 | 0.0000 |

Training result:

- frontier candidate rows: 16,910 train / 4,265 validation
- best validation top-1 frontier choice accuracy: 0.834
- online evaluation uses the frozen checkpoint; no training occurs during unseen-map exploration

## Current Baseline Recognized From The Project

The existing project implements goal-oriented exploration on unseen 2D grid maps:

- local observation only, no full map at inference time
- observed free/target cells are converted into a graph
- the policy predicts an action prior, not a full path
- a frontier selector chooses the next one-step action from the observed graph
- nearest-frontier is the baseline, utility-frontier is the proposed selector
- training uses supervised teacher traces, while online evaluation does not use goal coordinates or A*

The main limitation in the current slides is that the validation is still 2D. A second limitation is that utility can over-trust information gain and produce revisits or detours.

## Added 3D Direction

This update adds a reproducible 3D voxel version of the same research structure:

- map: layered `.vxl` voxel grid
- state: `(z, r, c)` robot pose
- actions: north, east, south, west, up, down, stop
- observation: local 3x3x3 partial voxel patch
- graph: observed free/target voxels connected by 6-neighbor edges
- frontier: observed free voxel adjacent to unknown voxel
- inference: one action at a time, followed by observation and graph update
- evaluation: hidden map is used only by the simulator for sensing/collision, not for planning

## Data Efficiency Hook

`train_voxel_policy.py` includes `--train-fraction`, so the same 3D experiment can directly compare full-data and low-data regimes:

- `--train-fraction 1.0`: full supervised data
- `--train-fraction 0.25`: 25 percent data
- `--train-fraction 0.10`: 10 percent data

This is a clean path for the next research contribution: show that partial-observation graph features plus an SSM-style policy keep goal discovery performance with less training data.

## Commands

Generate 3D maps:

```bash
python3 scripts/generate_voxel_maps.py --out-root maps --depth 5 --rows 12 --cols 12 --train 30 --test 10 --unseen 6
```

Build datasets:

```bash
python3 scripts/build_voxel_policy_dataset.py --map-dir maps/voxel_train --out datasets/voxel_nav/train.csv
python3 scripts/build_voxel_policy_dataset.py --map-dir maps/voxel_test --out datasets/voxel_nav/test.csv
```

Train with reduced data:

```bash
python3 scripts/train_voxel_policy.py \
  --train datasets/voxel_nav/train.csv \
  --val datasets/voxel_nav/test.csv \
  --out models/voxel_ssm_policy.pt \
  --log results/voxel_sim/train_log_fraction_025.csv \
  --epochs 20 \
  --hidden-dim 64 \
  --layers 2 \
  --train-fraction 0.25 \
  --cpu
```

Evaluate on unseen 3D maps:

```bash
python3 scripts/eval_voxel_frontier_modes.py \
  --maps maps/voxel_unseen/*.vxl \
  --model models/voxel_ssm_policy.pt \
  --modes nearest utility utility_commit \
  --out results/voxel_sim/frontier_modes_fraction_025
```

Create an interactive 3D viewer for one run:

```bash
python3 scripts/visualize_voxel_run.py \
  --map maps/voxel_unseen/unseen_001.vxl \
  --trajectory results/voxel_sim/frontier_modes_fraction_025/nearest/unseen_001/trajectory.csv \
  --observed results/voxel_sim/frontier_modes_fraction_025/nearest/unseen_001/observed_voxels.csv \
  --out results/voxel_sim/frontier_modes_fraction_025/nearest/unseen_001/voxel_3d_view.html
```

The generated viewer is a standalone HTML file. It can be opened directly in a browser and supports drag rotation, step-by-step playback, and layer toggles for obstacles, observed free voxels, unknown observed-state voxels, target, and trajectory.

## 3D Exploration Path Planner

`scripts/voxel_exploration_planner.py` is the direct path-planning entry point for the 3D task. It builds a 6-neighbor graph over observed free/target voxels and reconstructs a BFS path either to the hidden target in a fully known `.vxl` map or to a selected frontier in a partial observed voxel CSV.

Full-map target route:

```bash
python3 scripts/voxel_exploration_planner.py \
  --map maps/voxel_unseen/unseen_001.vxl \
  --mode target \
  --out results/voxel_sim/planner_check/target_path.csv
```

Partial-observation frontier route:

```bash
python3 scripts/voxel_exploration_planner.py \
  --map maps/voxel_unseen/unseen_001.vxl \
  --observed results/voxel_sim/rgbd_frontier/utility/unseen_001/observed_voxels.csv \
  --mode frontier \
  --selector utility \
  --out results/voxel_sim/planner_check/frontier_path.csv
```

The frontier selector supports `nearest` and `utility`. Utility favors unknown gain while penalizing travel distance and unnecessary z-layer changes. Output CSV rows contain `step,z,r,c,action_from_prev`, so the route can be reused by the evaluator or converted into a viewer trajectory.

## First Workspace Result

With the small checked experiment generated in this workspace:

- train rows: 445
- used rows with `--train-fraction 0.25`: 111
- validation action accuracy after 20 epochs: 0.562
- unseen 3D success rate:
  - nearest: 1.0
  - utility: 1.0
  - utility_commit: 0.833

This is not yet the final paper result. It is the first 3D pipeline validation. The current 3D utility still shows the same limitation as the 2D slide: if information gain dominates, it can detour or revisit. This should become the next improvement target rather than being hidden.

## Recommended Next Experiment

For the research story, prioritize data efficiency:

1. Train the same 3D model with fractions `1.0`, `0.5`, `0.25`, and `0.1`.
2. Evaluate each on the same `voxel_unseen` maps.
3. Report success rate, avg steps, observed nodes/edges, fallback count, and revisits.
4. Tune utility only after the data-efficiency trend is clear.

This creates a more differentiated claim than simply moving from 2D to 3D: the method performs 3D goal discovery in unseen maps while requiring less supervised navigation data.

## Camera-Based 3D Bridge

The first camera-based bridge is now `scripts/eval_rgbd_voxel_nav.py`.

Unlike the earlier voxel evaluator, it does not reveal a privileged local cube around the robot. Instead, it performs RGB-D/depth-style ray observations from the robot pose and updates only ray-visible voxels:

- free voxels along a depth ray become observed free space
- obstacle ray endpoints become observed obstacles
- target ray endpoints become observed target detections
- frontier selection sees only this reconstructed observed voxel graph

This is still a simulator, but it is a better intermediate step toward camera-based 3D VLN because the perception input is now closer to RGB-D/SLAM than to grid-world omniscient local observation.

Run:

```bash
python3 scripts/eval_rgbd_voxel_nav.py \
  --maps maps/voxel_unseen/*.vxl \
  --modes nearest utility \
  --out results/voxel_sim/rgbd_frontier \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2
```

The first workspace result:

- nearest: success rate 1.0, avg steps 21.50
- utility: success rate 1.0, avg steps 20.83

This should be treated as the first RGB-D/depth mapping sanity check, not the final VLN claim.

## Gazebo RGB-D Mapping Node

The next bridge is a ROS 2 node that consumes actual Gazebo depth camera topics:

- executable: `depth_voxel_mapper`
- launch: `src/s_nav_core/launch/rgbd_voxel_mapping.launch.py`
- depth input: `/camera/depth/image_raw`
- intrinsics input: `/camera/depth/camera_info`
- pose input: `/odom`
- output: `results/gazebo_rgbd/observed_voxels.csv`

Run after building:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select s_nav_core --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch s_nav_core rgbd_voxel_mapping.launch.py
```

This step moves from simulated depth rays in Python to real ROS depth images emitted by the Gazebo camera plugin. The current output is a runtime voxel map, not a full VLN policy yet.

The Gazebo launch files now use `worlds/small_house_world.sdf` by default instead of the earlier two-wall test world. It contains a compact house layout with two semantic target objects only: `target_red_chair` and a blue `bed`. The navigator projects the RGB-D voxel map to an observed free-space graph, selects frontiers from that graph, and stops when the requested target is visually confirmed at close range.

The dashboard exposes two semantic goals and a local LLM-style command box:

- chair: red chair target
- bed: blue bed target

Commands such as "미탐사 지역에서 빨간 의자를 찾아가" or "파란 침대 찾아" are grounded to those two targets. This is deterministic offline command grounding rather than a remote LLM call, which keeps the Gazebo validation repeatable before adding a learned detector or API-backed language module.

## Gazebo Experiment Dashboard

The dashboard script provides a single local control surface for Gazebo verification:

- start/stop `rgbd_voxel_mapping.launch.py`
- start/stop `rgbd_frontier_navigation.launch.py`
- check required ROS topics
- publish simple manual `/cmd_vel`
- monitor `results/gazebo_rgbd/observed_voxels.csv`
- inspect existing aggregate result files
- list generated 3D HTML viewers

`Start Mapping` verifies perception and voxel reconstruction only. `순찰 시작` or `LLM 명령으로 순찰 시작` runs the mapper plus `rgbd_frontier_navigator`, which reads the observed voxel map, extracts ground-projected frontiers, plans through the observed graph with BFS reachability/path reconstruction instead of A*, selects a coverage frontier, and publishes `/cmd_vel`.

The current Gazebo mission mode is `coverage_patrol`:

- patrol unexplored/reachable frontier regions in the unseen house map
- parse the local LLM-style command into a semantic target such as `chair` or `bed`
- keep exploring after a target sighting by default
- log target sightings to `results/gazebo_rgbd/target_events.csv`
- report `coverage_ratio`, `frontier_exhausted`, `target_event_count`, and `confirmed_target_count` in `results/gazebo_rgbd/metrics.csv`

This is intentionally different from a pure goal-seeking run. The validation target is whole-map patrol with event logging, not shortest-path navigation to a known goal.

Run:

```bash
python3 scripts/gazebo_experiment_dashboard.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

One-command launcher:

```bash
./run_gazebo_dashboard.sh nav
```

Inspect the live run:

```bash
cat results/gazebo_rgbd/metrics.csv
cat results/gazebo_rgbd/target_events.csv
tail -n 20 results/gazebo_rgbd/trajectory.csv
```

Interpretation:

- `mission_mode=coverage_patrol`: patrol mode is active.
- `coverage_ratio` increasing: the robot is expanding the observed map.
- `target_event_count > 0`: at least one target candidate/confirmed event was logged.
- `confirmed=1` in `target_events.csv`: the target was close, centered, and visually strong enough to count as confirmed.

This starts the dashboard and automatically launches Gazebo RGB-D frontier navigation. Use `mapping` instead of `nav` to verify only depth-to-voxel mapping, or `dashboard` to open only the UI.
