# Professor Demo Brief - 2026-05-21

## One-Line Message

2D grid 기반 SSM-Nav 검증을 3D voxel partial-observation 탐색으로 확장했고, Python RGB-D 시뮬레이션과 Gazebo RGB-D mapping/frontier patrol까지 이어지는 최소 동작 파이프라인을 만들었습니다.

## 보여줄 핵심

1. 3D voxel map에서 hidden full map 없이 observed voxel graph만 사용합니다.
2. RGB-D/depth ray로 관측된 free/occupied voxel을 누적합니다.
3. 관측된 free-space graph에서 frontier를 고르고 BFS path를 재구성합니다.
4. Gazebo에서는 depth mapper가 `observed_voxels.csv`를 만들고, navigator가 coverage patrol을 하며 chair/bed 목표 이벤트를 기록합니다.

## 구현된 산출물

- `scripts/voxel_exploration_planner.py`
  - 6-neighbor BFS 기반 3D target/frontier route planner
  - 출력: `step,z,r,c,action_from_prev`
- `scripts/eval_rgbd_voxel_nav.py`
  - privileged local cube 대신 RGB-D 스타일 ray observation으로 탐색 평가
- `scripts/visualize_voxel_run.py`
  - trajectory/observed voxel 3D HTML viewer 생성
  - 목표 voxel이 RGB-D 관측에 들어온 step부터 로봇-목표 시야선을 표시
- `src/s_nav_core/src/depth_voxel_mapper.cpp`
  - Gazebo depth image + odom + camera_info를 observed voxel CSV로 변환
- `src/s_nav_core/src/rgbd_frontier_navigator.cpp`
  - observed voxel CSV를 ground graph로 투영하고 frontier coverage patrol 수행
  - `metrics.csv`, `trajectory.csv`, `runtime_graph_nodes.csv`, `frontier_features.csv`, `target_events.csv` 기록
- `scripts/gazebo_experiment_dashboard.py`
  - Gazebo 실행, 토픽 확인, metric/trajectory/target event 확인용 로컬 대시보드

## 현재 결과 요약

Python RGB-D ray simulation, unseen voxel maps 6개:

| mode | success rate | avg steps success | avg revisit ratio |
| --- | ---: | ---: | ---: |
| nearest | 1.0 | 21.50 | 0.0083 |
| utility | 1.0 | 20.83 | 0.0079 |

3D voxel supervised policy fraction 실험도 실행되어 있습니다. 초기 결과 기준으로 full/low-data 조건 모두 성공률 비교가 가능하지만, utility weight와 commit 모드는 추가 튜닝 여지가 있습니다.

## ETPNav 대비 포지셔닝

ETPNav는 VLN-CE 환경에서 instruction, online topological map, cross-modal transformer planner, obstacle-avoiding controller를 결합한 계층형 navigation framework입니다. 핵심 공통점은 미리 주어진 전체 지도나 pre-exploration topology 없이, 이동 중 관측과 waypoint/topological node를 누적해 long-range planning에 쓰는 방향입니다.

현재 구현의 안전한 주장:

- ETPNav-inspired online topological memory/frontier exploration을 3D voxel partial-observation setting으로 재구성했습니다.
- Transformer cross-modal planner를 그대로 복제한 것이 아니라, 관측 graph/frontier feature와 SSM-style policy/utility selector를 결합하는 방향으로 연산 복잡도와 low-data 실험 가능성을 보려는 연구입니다.
- 현재 Python RGB-D demo의 성공은 target coordinate 도착이 아니라, RGB-D ray observation 안에 목표 voxel이 들어온 `target_seen_step` 기준입니다.

아직 과장하면 안 되는 주장:

- "ETPNav와 차이는 encoder만 Mamba로 바꾼 것"이라고 말하면 부족합니다. 현재는 task definition, metric, perception target, planner, evaluation domain도 다릅니다.
- `scripts/eval_rgbd_voxel_nav.py`의 RGB-D demo는 heuristic frontier utility 기반입니다. Mamba/SSM 쪽 주장은 `scripts/train_voxel_policy.py`와 `scripts/eval_voxel_frontier_modes.py` 결과와 연결해서 말해야 합니다.
- ETPNav의 R2R-CE/RxR-CE benchmark 성능과 현재 synthetic voxel map 성공률은 같은 수치로 직접 비교할 수 없습니다.

## 목요일 데모 순서

1. 연구 방향 설명
   - "2D grid에서 끝나는 것이 아니라 RGB-D 기반 3D partial observation으로 확장했습니다."
   - "온라인 탐색은 full map/goal coordinate를 직접 쓰지 않고 observed graph와 frontier만 봅니다."

1-1. 통합 연구 데모 실행

```bash
cd /home/mj/my_research/ssm_nav_ws
python3 scripts/run_exploration_research_demo.py \
  --command "if you find a red chair while exploring the unknown house, leave a log" \
  --out results/professor_demo/lightweight_vln_ssm_frontier \
  --limit-maps 6 \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2
```

이 데모는 자연어 명령을 local lightweight parser로 목표 spec으로 바꾸고, RGB-D ray observation으로 online voxel-frontier graph를 만들며, `utility`와 `ssm_utility` 정책을 비교 실행합니다. 현재 결과는 6개 unseen voxel map에서 두 모드 모두 target_seen 성공률 1.0입니다. 단, `ssm_utility`는 안전하지 않은 SSM action을 frontier utility fallback으로 보정하는 hybrid demo입니다.

확인할 파일:

```bash
cat results/professor_demo/lightweight_vln_ssm_frontier/demo_summary.txt
cat results/professor_demo/lightweight_vln_ssm_frontier/aggregate.csv
xdg-open results/professor_demo/lightweight_vln_ssm_frontier/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html
```

2. Python 3D path planner 확인

```bash
cd /home/mj/my_research/ssm_nav_ws
python3 scripts/voxel_exploration_planner.py \
  --map maps/voxel_unseen/unseen_001.vxl \
  --mode target \
  --out results/voxel_sim/planner_check/target_path.csv

python3 scripts/voxel_exploration_planner.py \
  --map maps/voxel_unseen/unseen_001.vxl \
  --observed results/voxel_sim/rgbd_frontier/utility/unseen_001/observed_voxels.csv \
  --mode frontier \
  --selector utility \
  --out results/voxel_sim/planner_check/frontier_path.csv
```

3. RGB-D 시뮬레이션 결과 재생성

```bash
python3 scripts/eval_rgbd_voxel_nav.py \
  --maps maps/voxel_unseen/*.vxl \
  --modes nearest utility \
  --out results/voxel_sim/rgbd_frontier \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2
```

4. 3D viewer 생성

```bash
python3 scripts/visualize_voxel_run.py \
  --map maps/voxel_unseen/unseen_001.vxl \
  --trajectory results/voxel_sim/rgbd_frontier/utility/unseen_001/trajectory.csv \
  --observed results/voxel_sim/rgbd_frontier/utility/unseen_001/observed_voxels.csv \
  --out results/voxel_sim/rgbd_frontier/utility/unseen_001/rgbd_3d_view.html
```

뷰어에서 `Full-map obstacles`와 `Target sight line`을 켜고 마지막 step으로 이동하면, 목표에 실제로 도착한 것이 아니라 RGB-D 관측에 들어온 순간을 선으로 보여줄 수 있습니다.

5. Gazebo dashboard 실행

```bash
./run_gazebo_dashboard.sh
```

브라우저:

```text
http://127.0.0.1:8765
```

대시보드가 먼저 뜨면 명령창에 예를 들어 `if you find a red chair patrolling the whole house, leave a log`를 입력하고 `자연어 명령 해석 후 시작`을 누릅니다. 드롭다운 목표를 그대로 쓰는 빠른 실행은 `선택 목표로 시작`입니다. 그 뒤 Gazebo가 켜지고, mapper가 먼저 voxel map을 만들며, 몇 초 뒤 navigator가 움직입니다.

확인할 파일:

```bash
cat results/gazebo_rgbd/metrics.csv
cat results/gazebo_rgbd/target_events.csv
tail -n 20 results/gazebo_rgbd/trajectory.csv
```

## 체크 기준

- ROS topic list에 다음 토픽이 보여야 합니다.
  - `/camera/image_raw`
  - `/camera/depth/image_raw`
  - `/camera/depth/camera_info`
  - `/odom`
- `observed_voxels.csv` 행 수가 시간에 따라 증가해야 합니다.
- `metrics.csv`에서 `mission_mode=coverage_patrol`이어야 합니다.
- `coverage_ratio`가 증가하면 map coverage patrol이 동작 중입니다.
- `target_event_count > 0`이면 chair/bed 후보 또는 확정 이벤트가 기록된 것입니다.

## 솔직히 말할 한계

- Gazebo navigator는 현재 3D voxel map을 ground-plane graph로 투영해서 patrol합니다. 즉, Python planner는 3D BFS이지만 Gazebo 주행은 differential-drive robot에 맞춘 2.5D 구현입니다.
- semantic target detector는 학습 모델이 아니라 색상 threshold 기반입니다. 현재 목표는 repeatable validation입니다.
- Gazebo RGB-D sensor는 rendering이 필요합니다. `DISPLAY`가 없으면 depth image가 안 나올 수 있습니다.
- 목요일 데모 전에는 topic fix 이후 Gazebo를 한 번 새로 돌려 최신 `metrics.csv`를 만들어두는 것이 좋습니다.

## 다음 연구 질문

1. Gazebo navigator를 ground projection이 아니라 3D voxel graph planner와 직접 연결할 수 있는가?
2. utility frontier의 revisit/detour를 줄이는 reward 또는 commitment 정책은 무엇이 좋은가?
3. low-data fraction에서 3D 탐색 성공률이 얼마나 유지되는가?
4. 색상 threshold 대신 object detector 또는 language-conditioned detector로 바꿨을 때 성능이 유지되는가?
