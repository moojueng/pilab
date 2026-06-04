# SSM-Nav-Onboard (C++ Implementation)

이 프로젝트는 Vision-Graph와 SSM(State Space Model)을 결합한 내비게이션 알고리즘을 C++ 환경에서 구현한 ROS 2 패키지입니다.

## 주요 구성 요소
- **s_nav_msgs**: 그래프 및 노드 상태를 위한 커스텀 메시지
- **s_nav_core**:
  - `GraphManager`: Vision-Graph 구축 및 관리
  - `SsmInference`: ONNX Runtime을 이용한 Mamba 모델 추론
  - `PathPlanner`: SSM 기반 경로 생성 정책

## 빌드 방법
```bash
cd ~/ssm_nav_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## 실행 방법
```bash
source install/setup.bash
ros2 run s_nav_core navigation_node --ros-args -p model_path:=/path/to/your/model.onnx
```

## 3D Voxel Research Extension

2D grid 검증을 3D voxel partial-observation navigation으로 확장하기 위한 Python 실험 파이프라인이 추가되었습니다.

- `scripts/generate_voxel_maps.py`: unseen 3D voxel map 생성
- `scripts/build_voxel_policy_dataset.py`: 3D local observation + graph feature dataset 생성
- `scripts/train_voxel_policy.py`: SSM-style policy 학습, `--train-fraction`으로 데이터 효율성 실험 지원
- `scripts/eval_voxel_frontier_modes.py`: nearest / utility / utility_commit 비교 평가
- `scripts/eval_rgbd_voxel_nav.py`: RGB-D/depth ray 관측으로 observed voxel map을 갱신하는 camera-based 3D 탐색 평가
- `scripts/voxel_exploration_planner.py`: observed voxel graph에서 6-neighbor BFS 기반 3D target/frontier 경로 계획
- `scripts/visualize_voxel_run.py`: 3D voxel map, observed voxels, trajectory를 standalone HTML로 시각화
- `docs/3d_voxel_research_note.md`: 현재 연구 해석, 실행 명령, 다음 실험 방향
- `docs/professor_demo_brief_2026-05-21.md`: 목요일 미팅용 핵심 요약, 데모 순서, 체크 기준
- `scripts/check_demo_outputs.py`: 데모 전 산출물/metric schema 점검

예시:

```bash
python3 scripts/generate_voxel_maps.py --out-root maps --depth 5 --rows 12 --cols 12 --train 30 --test 10 --unseen 6
python3 scripts/build_voxel_policy_dataset.py --map-dir maps/voxel_train --out datasets/voxel_nav/train.csv
python3 scripts/build_voxel_policy_dataset.py --map-dir maps/voxel_test --out datasets/voxel_nav/test.csv
python3 scripts/train_voxel_policy.py --train datasets/voxel_nav/train.csv --val datasets/voxel_nav/test.csv --out models/voxel_ssm_policy.pt --log results/voxel_sim/train_log_fraction_025.csv --epochs 20 --hidden-dim 64 --layers 2 --train-fraction 0.25 --cpu
python3 scripts/eval_voxel_frontier_modes.py --maps maps/voxel_unseen/*.vxl --model models/voxel_ssm_policy.pt --modes nearest utility utility_commit --out results/voxel_sim/frontier_modes_fraction_025
python3 scripts/visualize_voxel_run.py --map maps/voxel_unseen/unseen_001.vxl --trajectory results/voxel_sim/frontier_modes_fraction_025/nearest/unseen_001/trajectory.csv --observed results/voxel_sim/frontier_modes_fraction_025/nearest/unseen_001/observed_voxels.csv --out results/voxel_sim/frontier_modes_fraction_025/nearest/unseen_001/voxel_3d_view.html
```

3D voxel 경로 계획만 단독 확인:

```bash
python3 scripts/voxel_exploration_planner.py --map maps/voxel_unseen/unseen_001.vxl --mode target --out results/voxel_sim/planner_check/target_path.csv
python3 scripts/voxel_exploration_planner.py --map maps/voxel_unseen/unseen_001.vxl --observed results/voxel_sim/rgbd_frontier/utility/unseen_001/observed_voxels.csv --mode frontier --selector utility --out results/voxel_sim/planner_check/frontier_path.csv
```

RGB-D/depth ray 기반 3D 탐색:

```bash
python3 scripts/eval_rgbd_voxel_nav.py --maps maps/voxel_unseen/*.vxl --modes nearest utility --out results/voxel_sim/rgbd_frontier --max-steps 180 --depth-range 5 --aperture 2
python3 scripts/visualize_voxel_run.py --map maps/voxel_unseen/unseen_001.vxl --trajectory results/voxel_sim/rgbd_frontier/utility/unseen_001/trajectory.csv --observed results/voxel_sim/rgbd_frontier/utility/unseen_001/observed_voxels.csv --out results/voxel_sim/rgbd_frontier/utility/unseen_001/rgbd_3d_view.html
```

Gazebo RGB-D depth image를 실제 ROS topic으로 받아 voxel map을 갱신:

```bash
source install/setup.bash
ros2 launch s_nav_core rgbd_voxel_mapping.launch.py
```

출력:

```bash
results/gazebo_rgbd/observed_voxels.csv
```

Gazebo 실험 대시보드:

```bash
python3 scripts/gazebo_experiment_dashboard.py --host 127.0.0.1 --port 8765
```

브라우저에서:

```text
http://127.0.0.1:8765
```

대시보드에서 `Start Mapping`은 depth camera 기반 voxel map만 만들고, `순찰 시작`은 mapper와 `rgbd_frontier_navigator`를 함께 실행하여 coverage patrol을 시작합니다. 기본 동작은 목표를 발견해도 즉시 멈추지 않고 `results/gazebo_rgbd/target_events.csv`에 발견 이벤트를 남기며 계속 미탐사 frontier를 순찰하는 것입니다.

기본 Gazebo world는 `worlds/small_house_world.sdf`입니다. 거실/복도/욕실/침실/주방 구조와 `target_red_chair`, `bathroom_toilet`, bed, table, sofa가 들어 있습니다. 현재 자동 stop target은 RGB red detector 기반이라 `target_red_chair`를 찾는 검증에 맞춰져 있습니다.

대시보드 목표 선택:

- `빨간 의자 찾기`: red chair를 RGB 색상 단서로 찾음
- `침대가 있는 방으로 가기`: blue bed를 목표 단서로 찾음
- `화장실로 가기`: cyan bathroom/toilet marker를 목표 단서로 찾음

Gazebo depth camera는 렌더링이 필요합니다. 원격 서버에서 실행할 때는 MobaXterm X11 forwarding처럼 `DISPLAY`가 잡힌 터미널에서 실행해야 실제 depth image가 생성됩니다. `DISPLAY`가 없으면 Gazebo가 headless로 뜨며 depth camera가 비활성화되어 observed voxels가 0으로 남을 수 있습니다.

대시보드 먼저 실행:

```bash
./run_gazebo_dashboard.sh
```

브라우저가 열리면 명령어를 입력한 뒤 `자연어 명령 해석 후 시작`을 누릅니다. 드롭다운 목표를 그대로 쓰고 싶으면 `선택 목표로 시작`을 누릅니다. 이때 Gazebo, depth mapper, frontier navigator가 순서대로 시작됩니다. 예전처럼 바로 시작하고 싶으면 `./run_gazebo_dashboard.sh auto-nav`를 사용합니다.

실행 후 확인할 파일:

```bash
cat results/gazebo_rgbd/metrics.csv
cat results/gazebo_rgbd/target_events.csv
tail -n 20 results/gazebo_rgbd/trajectory.csv
```

확인 기준:

- `metrics.csv`의 `mission_mode`가 `coverage_patrol`이면 전체 순찰 모드로 실행 중입니다.
- `coverage_ratio`가 시간이 지나며 증가하면 미탐사 영역 순찰이 진행 중입니다.
- `target_event_count`가 1 이상이면 목표 후보/확정 발견 로그가 남은 것입니다.
- `target_events.csv`의 `confirmed=1`은 거리/중앙정렬/색상비 조건까지 만족한 확정 발견입니다.

데모 전 빠른 점검:

```bash
python3 scripts/check_demo_outputs.py
python3 scripts/check_demo_outputs.py --ros-topics
```

옵션:

```bash
./run_gazebo_dashboard.sh mapping
./run_gazebo_dashboard.sh dashboard
GOAL=bed ./run_gazebo_dashboard.sh auto-nav
```
