# LIMO ROS2 Half-Onboard Robot Architecture

작성일: 2026-06-08  
대상 학회: 추계전기학술대회  
전제: ICETA 2026 시뮬레이션 검증 결과를 실제 로봇 플랫폼으로 확장

## 1. 목표

본 문서는 LIMO ROS2 완제품 모바일로봇과 관제 PC를 이용하여, 시뮬레이션에서 검증한 SSM frontier graph 기반 탐색 시스템을 실제 로봇으로 확장하기 위한 기획/아키텍처 문서이다.

핵심 방향은 다음과 같다.

> 학습과 고수준 자연어/이상감지 cue 처리는 관제 PC에서 수행하고, 실제 주행 중 frontier graph 생성, SSM frontier scoring, 경로 추론, Nav2 주행은 LIMO ROS2 본체에서 수행한다.

이 구조는 완전 cloud/offboard가 아니라, 로봇 본체가 경로 판단을 수행하는 half-onboard architecture이다.

## 2. 시스템 구성

| 장치 | 역할 |
|---|---|
| LIMO ROS2 본체 | 센서 수집, odom/TF, 2D/3D map, frontier graph, SSM frontier scoring, Nav2 주행 |
| 관제 PC / 서버 | 학습, 데이터셋 구축, 모델 export, 자연어 명령 처리, 대형 모델 실험, RViz/로그 관리 |
| Wi-Fi 6 공유기 | LIMO ROS2와 관제 PC 간 ROS2/SSH/rsync 통신 |

예상 LIMO ROS2 구성:

| 항목 | 내용 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS | ROS2 Humble |
| compute | Intel NUC i7 |
| sensors | LiDAR, RGB-D camera, odom/IMU |
| navigation | Nav2, `cmd_vel`, TF tree |

## 3. 역할 분리

### 관제 PC에서 수행

| 기능 | 설명 |
|---|---|
| map/data collection 관리 | rosbag, 실험 로그, 학습 데이터 저장 |
| model training | SSM frontier scorer, anomaly model 학습 |
| model export | `.pt`, `.onnx`, `.yaml` 설정 파일 생성 |
| high-level language/anomaly cue | 자연어 명령을 target/anomaly/action condition으로 변환 |
| experiment dashboard | RViz, HTML dashboard, 결과 비교 |
| large model test | 필요 시 LLM/VLM/VLN 큰 모델 실험 |

### LIMO ROS2 본체에서 수행

| 기능 | 설명 |
|---|---|
| sensor bring-up | LiDAR, RGB-D, odom, TF |
| online map update | 2D occupancy / 3D voxel map 업데이트 |
| frontier graph update | 관측 free node와 unknown boundary 기반 frontier graph 생성 |
| SSM frontier scoring | 관제 PC에서 학습된 경량 scorer를 로봇에서 추론 |
| path inference | 가장 높은 점수의 frontier node를 subgoal로 선택 |
| navigation | Nav2 action 또는 `cmd_vel`로 실제 주행 |
| event logging | target/anomaly 발견 시 위치, 시간, 이미지/voxel 상태 기록 |

## 4. 전체 데이터 흐름

```text
1. 관제 PC
   - synthetic/real data 학습
   - SSM frontier scorer 학습
   - model checkpoint/export 생성

2. 자동 배포
   - rsync/scp로 model/config를 LIMO ROS2에 전송

3. LIMO ROS2
   - 센서 topic 수신
   - online 2D/3D map 업데이트
   - frontier 후보 생성
   - SSM frontier score 계산
   - best frontier node 선택
   - Nav2 goal 전송

4. 관제 PC
   - RViz/dashboard로 상태 확인
   - rosbag/log 수집
   - 필요 시 command/anomaly cue 업데이트
```

## 5. 네트워크 구성

```text
관제 PC Ubuntu 22.04
  - 학습 / 모델 export
  - 고수준 명령 처리
  - RViz / dashboard / 로그 수집
  - SSH / rsync / ROS2 action

        Wi-Fi 6 전용 공유기
        같은 subnet, 같은 ROS_DOMAIN_ID

LIMO ROS2 Ubuntu 22.04
  - sensor bring-up
  - map/frontier graph
  - SSM frontier scoring
  - Nav2 주행
```

권장 설정:

| 설정 | 권장값 |
|---|---|
| network | 연구실 전용 SSID |
| Wi-Fi band | 5GHz 우선 |
| IP | DHCP reservation 또는 static IP |
| ROS_DOMAIN_ID | 관제 PC와 LIMO 동일값 |
| ROS_LOCALHOST_ONLY | `0` |
| 배포 | SSH key + `rsync` |

예시:

```bash
export ROS_DOMAIN_ID=7
export ROS_LOCALHOST_ONLY=0
```

## 6. 자동 배포 계획

관제 PC에서 학습/수정한 모델을 LIMO ROS2로 자동 전송한다.

예상 파일:

```text
models/voxel_frontier_ssm.pt
config/frontier_nav.yaml
config/anomaly_cue.yaml
launch/limo_frontier_nav.launch.py
```

예시 명령:

```bash
rsync -av models/voxel_frontier_ssm.pt mj@LIMO_IP:~/ssm_nav_ws/models/
rsync -av config/ mj@LIMO_IP:~/ssm_nav_ws/config/
ssh mj@LIMO_IP "cd ~/ssm_nav_ws && source install/setup.bash && ros2 launch s_nav_core limo_frontier_nav.launch.py"
```

향후 만들 스크립트:

```text
scripts/deploy_to_limo.sh
scripts/run_limo_half_onboard_demo.sh
```

## 7. ROS2 노드 아키텍처

### LIMO ROS2 측 노드

| 노드 | 입력 | 출력 | 역할 |
|---|---|---|---|
| sensor drivers | camera/LiDAR/odom | `/scan`, `/camera/depth`, `/odom`, `/tf` | 센서 bring-up |
| voxel mapper | depth/RGB/TF | `/s_nav/voxel_map` | online 3D voxel map 생성 |
| frontier graph builder | voxel/occupancy map | `/s_nav/frontier_graph` | node/edge/frontier 후보 생성 |
| ssm frontier scorer | frontier features/model | `/s_nav/selected_frontier` | best frontier 선택 |
| nav goal bridge | selected frontier/TF | `/navigate_to_pose` | Nav2 goal 전송 |
| event logger | target/anomaly cue, observations | log/csv/rosbag | 발견 이벤트 저장 |

현재 관련 구현 파일:

```text
src/s_nav_core/src/depth_voxel_mapper.cpp
src/s_nav_core/src/rgbd_frontier_navigator.cpp
scripts/eval_rgbd_voxel_nav.py
scripts/voxel_frontier_features.py
scripts/train_voxel_frontier_ssm.py
```

### 관제 PC 측 노드/도구

| 도구 | 역할 |
|---|---|
| RViz2 | map, TF, path, goal 확인 |
| dashboard server | 실험 상태/결과 시각화 |
| command grounding node | 자연어 명령을 target/anomaly cue로 변환 |
| experiment runner | 실험 시작/종료, log sync |
| training pipeline | SSM/anomaly model 학습 |

## 8. 자연어/VLN 역할 정의

본 연구에서 VLN/LLM은 직접 경로를 생성하지 않는다.

| 항목 | 본 연구 역할 |
|---|---|
| 자연어 명령 | target/anomaly/action condition으로 변환 |
| 이상감지 cue | 어떤 관측을 이상으로 볼지 조건 생성 |
| 경로 추론 | LIMO ROS2 onboard frontier graph + SSM scorer가 수행 |

예시:

```text
"전체 집을 순찰하면서 빨간 의자를 발견하면 로그 남겨"
-> target_color: red
-> target_object: chair
-> mission_mode: coverage_patrol
-> on_detection_action: log
```

논문 표현:

> Language is used as a high-level semantic cue, while the actual exploration path is inferred onboard from the online frontier graph.

## 9. 시뮬레이션에서 로봇 구현으로 옮길 항목

| 시뮬레이션 구성 | 실제 로봇 대응 |
|---|---|
| voxel map `.vxl` | RGB-D/LiDAR 기반 online occupancy/voxel map |
| RGB-D ray sweep | 실제 depth camera point/depth stream |
| synthetic semantic target | RGB/RGB-D 기반 target/anomaly detector |
| frontier 후보 | 실제 map에서 unknown boundary |
| SSM frontier scorer | LIMO ROS2에서 model inference |
| trajectory csv | odom/Nav2 path/rosbag |
| HTML viewer | RViz2 + dashboard |

## 10. 실험 지표

추계전기학술대회에서 사용할 실제 로봇 지표:

| 지표 | 의미 |
|---|---|
| success rate | target/anomaly event 발견 여부 |
| target/anomaly seen step/time | 발견까지 걸린 step 또는 시간 |
| path length | 실제 이동 거리 |
| exploration coverage | 관측 map/voxel 비율 |
| revisit ratio | 반복 방문 비율 |
| frontier decision latency | 후보 선택 계산 시간 |
| Nav2 failure count | goal 실패/재시도 횟수 |
| CPU/RAM usage | LIMO NUC에서 onboard 추론 가능성 |
| Wi-Fi latency | 관제 PC와 통신 안정성 |

## 11. 구현 단계

### Phase 1. 네트워크/기본 주행

목표:

- LIMO ROS2와 관제 PC를 같은 Wi-Fi에 연결
- SSH 접속
- ROS2 topic list 확인
- RViz2에서 TF/LiDAR/depth/odom 확인
- Nav2 기본 goal 주행 확인

완료 조건:

```bash
ros2 topic list
ros2 topic echo /odom
ros2 run tf2_tools view_frames
```

### Phase 2. 모델 배포와 onboard inference

목표:

- 관제 PC에서 `models/voxel_frontier_ssm.pt`를 LIMO로 배포
- LIMO에서 feature 생성과 SSM scoring 실행
- frontier candidate별 점수 log 저장

완료 조건:

```text
frontier candidates generated
ssm scores generated
selected frontier published
decision latency logged
```

### Phase 3. 실제 센서 기반 frontier graph

목표:

- depth/LiDAR를 이용해 occupancy/voxel map 생성
- unknown boundary에서 frontier node 생성
- graph edge와 reachable frontier 확인

완료 조건:

```text
frontier graph visible in RViz/dashboard
selected frontier changes over time
robot path follows selected frontier
```

### Phase 4. 자연어/이상감지 cue 연동

목표:

- 관제 PC에서 자연어 명령 입력
- target/anomaly cue를 LIMO로 전달
- 발견 시 event log 생성

완료 조건:

```text
command -> cue -> onboard exploration -> detection event log
```

### Phase 5. 학회용 실험 정리

목표:

- 실제 로봇 3개 이상 환경에서 반복 실험
- utility vs ssm_utility 비교
- CPU/RAM/latency 측정
- 시뮬레이션 결과와 연결된 discussion 작성

## 12. 추계전기학술대회 주장 범위

안전한 주장:

- LIMO ROS2 기반 실제 로봇에서 online frontier graph 탐색 구현
- 관제 PC 학습 모델을 LIMO ROS2에 배포하여 onboard SSM frontier scoring 수행
- 자연어 명령/이상감지 cue는 고수준 조건으로 사용
- 실제 경로 추론과 주행은 로봇 본체에서 수행
- 시뮬레이션으로 검증한 구조가 실제 ROS2/Nav2 플랫폼으로 확장 가능함을 보임

피해야 할 주장:

- 대형 VLM/VLN을 LIMO ROS2 본체에서 완전 실시간 실행했다는 주장
- MTU3D/ETPNav 공식 벤치마크를 능가했다는 주장
- 모든 환경에서 최적 경로를 보장한다는 주장
- 완전 autonomous anomaly understanding이 끝났다는 주장

권장 표현:

> We implement a half-onboard robot system where the LIMO ROS2 platform performs onboard frontier-graph path inference using a lightweight SSM scorer, while the control PC handles training and high-level language/anomaly cue generation.

한국어:

> 본 연구는 관제 PC가 학습과 고수준 자연어/이상감지 cue 생성을 담당하고, LIMO ROS2 본체가 online frontier graph와 경량 SSM scorer를 이용해 실제 경로 추론과 주행을 수행하는 하프 온보드 모바일로봇 시스템을 구현한다.

