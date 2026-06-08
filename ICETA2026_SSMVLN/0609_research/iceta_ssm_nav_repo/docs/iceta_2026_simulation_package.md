# ICETA 2026 Simulation Package

작성일: 2026-06-08  
목표 제출일: 2026-06-13  
우선순위: 시뮬레이션 기반 학회 발표 자료 완성

## 1. 발표 핵심 주제

본 연구는 미지 공간에서 이상감지/목표 발견을 수행하는 모바일로봇 탐색 시스템이다. 대형 VLN 모델이 직접 경로를 생성하는 방식이 아니라, 자연어 명령과 이상감지 cue를 탐색 조건으로 사용하고, 실제 경로 추론은 로봇 내부의 online frontier graph와 SSM 기반 frontier scorer가 수행하는 구조이다.

ICETA 발표에서는 다음 문장을 중심 주장으로 둔다.

> We propose a lightweight language-conditioned 2D/3D frontier-graph exploration framework where a one-time-trained SSM-style scorer selects informative frontier nodes from online RGB-D observations.

한국어 발표 표현:

> 본 연구는 자연어 명령을 탐색 조건으로 변환한 뒤, RGB-D 기반 online 3D voxel frontier graph와 1회 학습된 경량 SSM frontier scorer를 이용해 미지 공간을 효율적으로 탐색하는 방법이다.

중요한 구분:

| 항목 | 역할 |
|---|---|
| `ssm_utility` | 본 연구의 시뮬레이션 제안 방식 |
| `mtu3d_proxy` | MTU3D의 active frontier-query 탐색 개념을 local simulator 신호로 근사한 비교논문식 proxy |
| `utility` | 내부 ablation/control용 hand-crafted frontier heuristic |
| MTU3D | 비교논문. 공식 코드/체크포인트 benchmark 재현은 아직 아님 |

따라서 ICETA 결과는 두 층으로 제시한다.

1. **Simulation validation / paper-proxy comparison:** `ssm_utility`가 미지 3D voxel map에서 동작하고, `mtu3d_proxy` 대비 어떤 변화가 있는지 확인한다.
2. **Paper comparison:** MTU3D 공식 benchmark를 재현한 것은 아니므로, dashboard 수치는 `MTU3D-style Proxy` 비교로 표기하고, 논문 자체와는 문제정의, 지도표현, 학습방식, 경로결정, 배치 가능성을 방법론적으로 비교한다.

## 2. 현재 구현 상태

| 구분 | 구현 상태 | 관련 파일 |
|---|---|---|
| 3D voxel map 생성 | 구현됨 | `scripts/generate_voxel_maps.py` |
| semantic target 삽입 | 구현됨 | `scripts/ensure_semantic_voxel_targets.py` |
| frontier 후보 데이터셋 구축 | 구현됨 | `scripts/build_voxel_frontier_dataset.py` |
| SSM-style frontier scorer 학습 | 구현됨 | `scripts/train_voxel_frontier_ssm.py` |
| RGB-D partial observation 시뮬레이션 | 구현됨 | `scripts/eval_rgbd_voxel_nav.py` |
| 자연어 명령 grounding | 구현됨, symbolic/local LLM 선택 가능 | `scripts/local_vln_llm.py` |
| 3D 주행 viewer | 구현됨 | `scripts/visualize_voxel_run.py` |
| 통합 dashboard | 구현됨 | `scripts/build_voxel_study_dashboard.py`, `scripts/voxel_study_dashboard_server.py` |
| 비교논문 정리 | 구현됨 | `docs/mtu3d_comparison_note.md` |

## 3. 학습 모델

### 모델명

현재 학습 모델은 `VoxelFrontierSsmNet`이다.

위치:

```text
scripts/train_voxel_frontier_ssm.py
```

### 모델 구조

| 구성 | 내용 |
|---|---|
| 입력 | frontier 후보별 feature vector |
| hidden dim | 기본 96 |
| layer 수 | 기본 3 |
| block | gated MLP + residual + LayerNorm 형태의 SSM-style block |
| 출력 | 해당 frontier 후보의 scalar score |
| 선택 방식 | 각 step의 frontier 후보 중 score가 가장 높은 후보를 다음 subgoal로 선택 |

코드상 핵심 구조:

```text
input frontier feature
-> Linear projection
-> SsmBlock x 3
-> score head
-> frontier score
```

### 입력 feature

입력 feature는 `scripts/voxel_frontier_features.py`의 `FRONTIER_FEATURE_NAMES`를 따른다. 주요 정보는 다음과 같다.

| feature 종류 | 의미 |
|---|---|
| robot/frontier 상대 위치 | 현재 로봇 위치와 후보 frontier의 3D 상대 관계 |
| graph distance | observed graph에서 해당 frontier까지 거리 |
| first action | 후보 frontier로 가기 위한 첫 이동 방향 |
| observed nodes/edges | 현재까지 관측된 위상/voxel graph 규모 |
| frontier count | 현재 step의 후보 frontier 수 |
| unknown count | 후보 주변 미탐사 voxel 밀도 |
| free degree | 후보 주변의 관측 free 연결성 |
| visit penalty | 반복 방문 가능성/패널티 |

### 학습 목표

학습 target은 후보 frontier가 얼마나 좋은 탐색 후보인지 나타내는 `target_score`이다. 이 score는 숨겨진 free-space gain, unknown gain, target prior, travel cost, revisit penalty, dead-end penalty, vertical penalty를 조합해 생성한다.

즉, 학습은 다음 문제로 정의된다.

> 현재 online 관측 graph에서 여러 frontier 후보가 있을 때, 어느 후보가 이후 탐색과 목표 발견에 더 유리한지 점수화하는 ranking/regression 문제

## 4. 학습 방법

### 데이터셋 생성

훈련/검증용 voxel map을 생성하고, 각 map에서 oracle path를 따라가며 step별 frontier 후보 row를 만든다.

대표 명령:

```bash
cd /home/mj/my_research/ssm_nav_ws

python3 scripts/generate_voxel_maps.py \
  --out-root maps/vln_ssm_voxel_study \
  --depth 5 --rows 12 --cols 12 \
  --train 40 --test 10 --unseen 6 \
  --seed 710

python3 scripts/build_voxel_frontier_dataset.py \
  --map-dir maps/vln_ssm_voxel_study/voxel_train \
  --out datasets/voxel_frontier_study/train_frontiers.csv

python3 scripts/build_voxel_frontier_dataset.py \
  --map-dir maps/vln_ssm_voxel_study/voxel_test \
  --out datasets/voxel_frontier_study/test_frontiers.csv
```

현재 데이터셋 크기:

| split | rows |
|---|---:|
| train frontier rows | 16,910 |
| validation frontier rows | 4,265 |

파일 기준으로는 header를 포함해 각각 `16911`, `4266` lines이다.

### SSM frontier scorer 학습

대표 명령:

```bash
cd /home/mj/my_research/ssm_nav_ws

python3 scripts/train_voxel_frontier_ssm.py \
  --train datasets/voxel_frontier_study/train_frontiers.csv \
  --val datasets/voxel_frontier_study/test_frontiers.csv \
  --out models/voxel_frontier_ssm.pt \
  --log results/vln_ssm_voxel_study/train_frontier_ssm_log.csv \
  --epochs 50 \
  --seed 710
```

현재 학습 결과:

| 항목 | 값 |
|---|---:|
| best validation top-1 frontier choice accuracy | 0.834 |
| model checkpoint | `models/voxel_frontier_ssm.pt` |
| train log | `results/vln_ssm_voxel_study/train_frontier_ssm_log.csv` |

발표 표현:

> The SSM-style scorer is trained once on synthetic frontier candidates and then reused on unseen maps without per-map retraining.

## 5. 시뮬레이션 실행

ICETA 발표에서는 semantic-prior target 배치가 적용된 12-map 결과를 1차 발표 산출물로 사용한다.

```text
results/professor_demo/iceta_semantic_prior
```

해당 결과의 dashboard도 생성해두었다.

```text
results/professor_demo/iceta_semantic_prior/dashboard.html
```

재현/추가 확인용으로 ICETA 이름이 들어간 별도 실행 결과도 생성해두었다.

```text
results/iceta_2026_simulation
```

발표 우선 결과 재실행 명령:

```bash
cd /home/mj/my_research/ssm_nav_ws

python3 scripts/run_exploration_research_demo.py \
  --command "if you find a red chair while exploring the unknown house, leave a log" \
  --map-dir maps/iceta_semantic_prior/voxel_unseen \
  --model models/iceta_semantic_prior_ssm.pt \
  --modes mtu3d_proxy ssm_utility \
  --out results/professor_demo/iceta_semantic_prior \
  --limit-maps 12 \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2 \
  --hybrid-ssm-weight 0.35 \
  --ssm-candidate-window 0.12 \
  --ssm-override-margin 0.08 \
  --llm-provider symbolic

python3 scripts/build_voxel_study_dashboard.py \
  --result-dir results/professor_demo/iceta_semantic_prior \
  --map-dir maps/iceta_semantic_prior/voxel_unseen \
  --out results/professor_demo/iceta_semantic_prior/dashboard.html
```

추가 확인용 ICETA package 재실행 명령:

```bash
cd /home/mj/my_research/ssm_nav_ws

python3 scripts/run_vln_ssm_voxel_study.py \
  --command "전체 집을 순찰하면서 빨간 의자를 발견하면 로그 남겨" \
  --out results/iceta_2026_simulation \
  --unseen-maps 6 \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2 \
  --llm-provider symbolic
```

위 명령은 다음 과정을 한 번에 수행한다.

1. voxel train/test/unseen map 확인 또는 생성
2. unseen map에 semantic target 삽입 확인
3. 기존 `models/voxel_frontier_ssm.pt` 로드
4. 자연어 명령을 `red_chair`, `log` mission spec으로 grounding
5. 논문 proxy 비교를 위해 `mtu3d_proxy`와 `ssm_utility` 모드 평가
6. `aggregate.csv`, `summary.csv`, 개별 `metrics.csv`, 3D viewer, dashboard 생성

## 6. 대시보드 확인

### 발표 우선 dashboard 열기

```bash
cd /home/mj/my_research/ssm_nav_ws
xdg-open results/professor_demo/iceta_semantic_prior/dashboard.html
```

### 발표 우선 개별 3D viewer 열기

```bash
cd /home/mj/my_research/ssm_nav_ws
xdg-open results/professor_demo/iceta_semantic_prior/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html
```

### 추가 확인용 dashboard 열기

```bash
cd /home/mj/my_research/ssm_nav_ws
xdg-open results/iceta_2026_simulation/dashboard.html
```

### 추가 확인용 개별 3D viewer 열기

```bash
cd /home/mj/my_research/ssm_nav_ws
xdg-open results/iceta_2026_simulation/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html
```

### 명령어 입력형 dashboard 서버 실행

```bash
cd /home/mj/my_research/ssm_nav_ws

python3 scripts/voxel_study_dashboard_server.py \
  --host 127.0.0.1 \
  --port 8787 \
  --result-dir results/iceta_2026_simulation \
  --map-root maps/vln_ssm_voxel_study \
  --dataset-root datasets/voxel_frontier_study \
  --model models/voxel_frontier_ssm.pt \
  --llm-provider symbolic \
  --unseen-maps 6
```

브라우저:

```text
http://127.0.0.1:8787
```

### 대시보드에서 확인할 것

| 화면 요소 | 확인 내용 | 발표 의미 |
|---|---|---|
| Natural Language Command | 자연어 명령 입력과 grounding 결과 | VLN/LLM은 경로 생성이 아니라 mission cue 생성 |
| Paper-Proxy Comparison | `MTU3D-style Proxy` vs `Proposed SSM` 비교 | 공식 MTU3D 재현이 아니라 local simulator proxy 비교 |
| 3D voxel scene | 장애물, 관측 free voxel, target, trajectory | partial RGB-D 관측 기반 online 탐색 증명 |
| Timeline slider | step별 이동/관측 변화 | 미지 공간에서 점진적으로 graph 확장 |
| Map Runs table | map별 success, target step, revisit, nodes | unseen map generalization |
| Case status | fallback, frontier switch, depth rays | 안전 fallback과 frontier 선택 동작 |

3D viewer 색상 해석:

| 색상/레이어 | 의미 |
|---|---|
| black/dark voxel | obstacle |
| white/transparent voxel | observed free voxel |
| red voxel | target object |
| blue path | trajectory |
| green point | current robot |
| orange line | target sight line |

## 7. 현재 ICETA 결과: Simulation Validation / Paper-Proxy Comparison

### 1차 발표 추천 결과

파일:

```text
results/professor_demo/iceta_semantic_prior/aggregate.csv
results/professor_demo/iceta_semantic_prior/demo_summary.txt
results/professor_demo/iceta_semantic_prior/dashboard.html
```

12개 unseen voxel map 결과:

| mode | success | avg successful target step | avg revisit ratio | avg decision ms | avg frontiers |
|---|---:|---:|---:|---:|---:|
| mtu3d_proxy | 0.917 | 38.27 | 0.1763 | 54.57 | 80.07 |
| ssm_utility | 1.000 | 22.08 | 0.0139 | 42.39 | 79.94 |
| hybrid_ssm | 1.000 | 33.67 | 0.0770 | 97.19 | 80.01 |

해석:

- `ssm_utility`는 12개 unseen map에서 target observation success를 91.7%에서 100%로 높였다.
- 성공한 episode 기준 평균 target step은 38.27에서 22.08로 줄었고, revisit ratio도 0.1763에서 0.0139로 낮췄다.
- 현재 local prototype에서는 독립 `ssm_utility`의 평균 decision time도 `mtu3d_proxy`보다 낮았다. 단, 이 수치는 Python simulator 구현 기준이므로 실제 onboard runtime 우위는 Jetson/LIMO 실험에서 별도로 검증한다.
- `hybrid_ssm`은 기존 proxy 후보를 SSM으로 re-ranking한 ablation이다. 발표 주 비교는 proxy를 호출하지 않는 독립 `ssm_utility`와 `mtu3d_proxy` 사이로 둔다.
- 여기서 `mtu3d_proxy`는 비교논문 MTU3D의 공식 재현이 아니라, local voxel simulator에서 active frontier-query 관점을 근사한 proxy이다.
- 이 표는 MTU3D 공식 benchmark 성능표가 아니라, 동일 simulator 조건에서의 paper-proxy comparison 결과이다.

주의:

- 이 결과의 `mission.json`은 `llm_used=false`, `grounding_type=offline_symbolic_fallback`이다.
- 따라서 발표에서는 "local symbolic/VLN-command grounding" 또는 "natural-language command grounding prototype"으로 표현한다.
- 실제 local HF LLM을 사용한 결과가 필요하면 `voxel_study_dashboard_server.py`를 `--llm-provider hf_local`로 실행해 별도 결과를 만든다.

### Historical Internal Ablation Reference

아래 결과는 이전 `utility` control과의 내부 ablation 기록이다. 현재 발표 주 표는 위의 `mtu3d_proxy` vs 독립 `ssm_utility` 결과를 사용한다.

파일:

```text
results/iceta_2026_simulation/aggregate.csv
results/iceta_2026_simulation/demo_summary.txt
```

구버전 6개 unseen voxel map 결과:

| mode | success | avg target step | avg revisit ratio | avg decision ms | avg frontiers |
|---|---:|---:|---:|---:|---:|
| utility | 1.000 | 18.17 | 0.0610 | 4.64 | 81.33 |
| ssm_utility | 1.000 | 19.33 | 0.0300 | 43.51 | 79.23 |

해석:

- 두 모드 모두 6개 unseen map에서 target observation success 100%를 보였다.
- `ssm_utility`는 평균 target step은 `utility`보다 약간 길지만, revisit ratio를 약 0.0610에서 0.0300으로 낮췄다.
- 이 구버전 내부 ablation에서는 SSM decision latency가 `utility`보다 컸다. 현재 주 결과에서는 `mtu3d_proxy` 대비 독립 `ssm_utility`의 simulator decision time이 더 낮게 측정되었지만, 실제 onboard runtime은 별도 검증한다.
- 전체 발표 강조점은 "모든 실행에서 모든 metric 압도"가 아니라 "자연어 cue + explicit frontier graph + one-time-trained SSM scorer를 결합한 탐색 구조"이다.
- `utility` 결과는 논문 baseline 재현 결과가 아니라 내부 ablation/control result로만 사용한다.

## 8. 비교논문과 차이점: MTU3D Method-Level Comparison

ICETA에서 비교논문으로 사용할 논문:

```text
MTU3D: Move to Understand a 3D Scene:
Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation
2025, ICCV 2025 Highlight / arXiv preprint
```

세부 정리는 다음 파일에 있다.

```text
docs/mtu3d_comparison_note.md
```

주의:

- 현재 dashboard에는 MTU3D가 직접 구현되어 있지 않다.
- 따라서 `utility` 결과를 MTU3D 결과처럼 설명하면 안 된다.
- 같은 benchmark에서 MTU3D 공식 코드와 본 연구 코드를 같이 돌리지 않았으므로, "MTU3D보다 성능이 좋다"는 수치 주장은 하지 않는다.
- 대신 본 연구가 MTU3D와 다른 연구 방향을 갖는다는 점, 즉 explicit graph와 lightweight SSM scorer 기반의 deployable exploration이라는 점을 비교한다.

### 방법론 비교 요약

| 항목 | MTU3D | 본 연구 |
|---|---|---|
| 핵심 목적 | 3D visual grounding + embodied navigation | 이상감지/목표 발견 중심의 unknown-space exploration |
| 언어 사용 | 3D scene grounding과 navigation 목표 이해 | 자연어 명령을 target/anomaly/action cue로 변환 |
| 지도 표현 | learned 3D VL memory/query | explicit online 2D/3D frontier graph |
| frontier 표현 | learned frontier queries | 관측 free voxel/cell과 unknown 경계의 명시적 node |
| 모델 | 큰 3D vision-language-exploration model | local grounding + lightweight SSM frontier scorer |
| 학습 | 대규모 3D-VL/exploration pretraining/fine-tuning | synthetic frontier 후보로 SSM scorer 초기 1회 학습 |
| online 탐색 | learned model 기반 exploration | graph update, frontier scoring, path planning 분리 |
| onboard 방향 | 경량 onboard 검증이 핵심 주장 아님 | LIMO ROS2/Jetson까지 염두에 둔 경량 inference 구조 |

### 공정한 실험 비교를 하려면 필요한 것

MTU3D와 수치 비교를 하려면 아래 중 하나가 필요하다.

| 방법 | 설명 | ICETA 현재 상태 |
|---|---|---|
| MTU3D official benchmark 재현 | HM3D-OVON, GOAT-Bench, SG3D, A-EQA 등에서 MTU3D 공식 코드와 본 연구를 같은 조건으로 평가 | 아직 미수행 |
| 본 연구 simulator에 MTU3D adapter 구현 | MTU3D의 RGB-D input, memory/query, frontier selection을 현재 voxel simulator에 맞춰 변환 | 아직 미수행 |
| reported result comparison | MTU3D 논문 보고 수치와 본 연구 시뮬레이션 수치를 나란히 두되, dataset/protocol이 다름을 명시 | 가능하지만 직접 우위 주장 불가 |

ICETA에서는 첫 제출까지 시간이 짧으므로, **정량 결과는 내부 ablation으로 제시하고, MTU3D는 방법론 비교논문으로 제시**하는 것이 안전하다.

### 발표에서 강조할 더 나은 점

1. **Exploration-first**
   - 목표물로 최단 이동하는 VLN이 아니라, 미지 공간을 탐색하면서 anomaly/target cue를 발견하는 구조이다.

2. **Explicit 2D/3D frontier graph**
   - 학습 모델 안에 모든 공간 기억을 숨기지 않고, online graph로 node/edge를 명확히 유지한다.
   - 추후 실제 로봇에서 ROS2/Nav2와 연결하기 쉽다.

3. **One-time training, unseen map inference**
   - map마다 재학습하지 않고, 초기 학습된 frontier scorer를 unseen map에서 재사용한다.

4. **Lightweight deployment path**
   - 대형 VLN/VLM이 직접 action을 뽑는 구조가 아니라, language/anomaly cue와 경로 추론을 분리한다.
   - LIMO ROS2 본체에서 frontier graph/SSM path inference가 가능하고, 관제 PC는 학습/고수준 cue 처리로 분리할 수 있다.

5. **Dashboard-visible validation**
   - 논문/발표에서 정량 결과뿐 아니라 step별 관측 voxel, target sighting, trajectory를 직접 보여줄 수 있다.

## 9. 발표 슬라이드 구성안

1. Problem
   - 미지 환경에서 이상감지/목표 발견을 위한 모바일로봇 탐색 필요

2. Limitation of Existing VLN/Object Navigation
   - 기존 VLN은 지정된 목표로 이동하는 데 초점
   - exploration-first anomaly discovery에는 별도 구조가 필요

3. Proposed System
   - natural-language cue
   - RGB-D partial observation
   - online 3D voxel frontier graph
   - SSM frontier scorer
   - graph/Nav2-style path planning

4. Training Pipeline
   - synthetic voxel maps
   - frontier candidate dataset
   - VoxelFrontierSsmNet one-time training

5. Simulation Setup
   - 5-layer 12x12 voxel maps
   - unseen maps
   - RGB-D ray sweep
   - internal control: utility
   - proposed mode: ssm_utility

6. Results
   - success rate
   - target seen step
   - revisit ratio
   - decision latency
   - 이 결과는 내부 ablation이며 MTU3D 재현 결과가 아님

7. Dashboard Demonstration
   - `dashboard.html`
   - `rgbd_ssm_frontier_view.html`

8. Comparison With MTU3D / ETPNav
   - MTU3D: method-level 3D vision-language exploration comparison
   - ETPNav: VLN/topological navigation baseline
   - 본 연구: exploration-first anomaly-aware frontier graph

9. Real-Robot Extension
   - LIMO ROS2 + control PC half-onboard architecture

10. Conclusion and Next Steps
   - ONNX/C++ optimization
   - real RGB-D/LiDAR integration
   - LIMO ROS2 deployment

## 10. 6월 13일까지 남은 작업

| 날짜 | 작업 | 산출물 |
|---|---|---|
| 6/8 | ICETA 결과 패키지 생성 | `results/iceta_2026_simulation` |
| 6/9 | 발표용 그림 캡처 | dashboard, 3D viewer screenshot |
| 6/10 | abstract/introduction/method 정리 | 학회 원고 초안 |
| 6/11 | results/comparison 문장 정리 | table, figure caption |
| 6/12 | 발표 슬라이드/원고 최종 확인 | PDF, demo link |
| 6/13 | 제출 | final package |

## 11. 안전한 발표 문장

과장 없이 쓰기 좋은 문장:

> The current simulation does not claim to outperform large-scale 3D vision-language models on their benchmarks. Instead, it validates a lightweight and modular exploration pipeline where language/anomaly cues are separated from onboard frontier-graph path inference.

한국어:

> 현재 시뮬레이션은 대형 3D vision-language 모델의 벤치마크 성능을 직접 능가한다고 주장하지 않는다. 대신 자연어/이상감지 cue와 로봇의 onboard frontier graph 기반 경로 추론을 분리한 경량 탐색 구조의 가능성을 검증하는 데 초점을 둔다.

MTU3D 비교용 안전 문장:

> MTU3D는 visual grounding과 active exploration을 통합한 대형 3D vision-language-exploration 모델인 반면, 본 연구는 미지 공간 탐색을 explicit online frontier graph와 경량 SSM scorer로 분리하여 실제 모바일로봇 배치 가능성을 높이는 방향에 초점을 둔다.

local paper-proxy 비교용 안전 문장:

> 본 연구의 시뮬레이션에서는 MTU3D 공식 benchmark를 직접 재현하지 않았으며, 동일 local semantic-prior voxel simulator 안에서 MTU3D-style frontier-query proxy와 Proposed SSM mode를 비교하였다.
