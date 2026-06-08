# ICETA Dashboard Experiment Checklist

작성일: 2026-06-08  
대상 화면: `results/professor_demo/iceta_semantic_prior/dashboard.html`

## 0. 먼저 구분할 것

이 dashboard는 비교논문 MTU3D의 공식 코드/체크포인트를 재현한 화면이 아니다. 대신 현재 local voxel simulator 안에서 `MTU3D-style Proxy`와 본 연구 `Proposed SSM`을 같은 map 조건으로 비교하도록 구성했다.

| 구분 | 역할 |
|---|---|
| `ssm_utility` | 본 연구의 시뮬레이션 제안 방식 |
| `mtu3d_proxy` | MTU3D의 active frontier-query 탐색 개념을 local simulator 신호로 근사한 비교논문식 proxy |
| `utility` | 내부 ablation/control용 hand-crafted frontier heuristic. 현재 발표용 주 비교축은 아님 |
| MTU3D | 비교논문. 공식 benchmark 재현은 아직 아님 |

따라서 dashboard에서 확인하는 것은 "제안 방식이 미지 3D 공간에서 동작하는가"와 "MTU3D-style proxy 대비 target 발견 step, revisit, 관측 graph 확장량이 어떻게 달라지는가"이다. 발표에서는 반드시 `MTU3D-style Proxy`가 공식 MTU3D 재현이 아니라 local simulator proxy라고 명시한다.

## 1. 대시보드에서 봐야 하는 것

### Study 영역

| 항목 | 현재 의미 | 주장 연결 |
|---|---|---|
| `100.0% SSM success` | 선택된 SSM 계열 모드가 unseen map에서 target을 관측함 | 미지 환경에서도 target/anomaly cue 기반 탐색 가능 |
| `22.08 SSM avg target step` | target을 관측하기까지 평균 successful step | 제안 방식의 시뮬레이션 동작 확인 |
| `frontier top-1` | 학습 log 연결 시 frontier 선택 accuracy 표시 | 학습된 scorer가 frontier ranking을 배움 |
| `12 unseen maps` | 평가에 사용한 unseen map 수 | 학습 map이 아닌 환경에서 평가 |

정량 결과는 `aggregate.csv`를 우선 인용한다.

### Mode 영역

| mode | 의미 |
|---|---|
| `mtu3d_proxy` / `MTU3D-style Proxy` | 비교논문 MTU3D의 frontier-query active exploration 관점을 local voxel simulator에 맞춘 proxy |
| `ssm_utility` / `Proposed SSM` | MTU3D-style proxy를 호출하지 않는 독립 target-conditioned SSM frontier selection |
| `hybrid_ssm` / `Hybrid SSM` | MTU3D-style proxy 후보를 SSM으로 re-ranking하는 ablation |
| `utility` / `Utility Control` | 내부 heuristic frontier selection mode. 비교논문 방식이 아님 |

발표에서는 `utility`를 "비교논문 baseline"이라고 부르지 않는다. 현재 dashboard의 논문 비교용 버튼은 `MTU3D-style Proxy`이고, 본 연구 버튼은 `Proposed SSM`이다.

### Current Case 영역

| 항목 | 봐야 할 것 | 해석 |
|---|---|---|
| `grounding` | `offline_symbolic_fallback` | 현재는 local symbolic command grounding 결과 |
| `llm_used` | `false` | 대형 LLM 실험이 아니라 command grounding prototype |
| `fallbacks` | 0이면 좋음 | SSM 선택이 안전하지 않아 fallback한 횟수 |
| `frontier switches` | 너무 크지 않은지 | 탐색 목표가 자주 흔들리는지 확인 |
| `depth rays` | RGB-D 관측량 | partial observation 기반임을 설명 |

### Case Metrics 영역

| 항목 | 주장 연결 |
|---|---|
| `success=yes` | target/anomaly cue 발견 성공 |
| `target seen step` | 발견 속도 |
| `revisit ratio` | 반복 방문 억제 |
| `observed nodes` | online graph가 얼마나 확장됐는지 |

### 3D Scene 영역

| 시각 요소 | 의미 |
|---|---|
| 검은 voxel | 장애물 |
| 흰/투명 voxel | 관측된 free space |
| 빨간 target | command target |
| 파란 trajectory | 로봇 이동 경로 |
| 초록 point | 현재 로봇 |
| 주황 sight line | target이 관측된 시점 |

이 화면에서는 timeline slider를 움직이며 "로봇이 미리 전체 map을 아는 것이 아니라, RGB-D 관측으로 graph를 확장하면서 target을 발견한다"는 점을 보여준다.

### Paper-Proxy Comparison 영역

현재 발표 추천 결과:

| mode | avg target step | revisit |
|---|---:|---:|
| MTU3D-style Proxy | 38.27 | 17.6% |
| Proposed SSM | 22.08 | 1.4% |

주장:

> Local voxel simulator에서 MTU3D-style frontier-query proxy와 비교했을 때, 제안한 SSM-frontier graph 방식은 동일 unseen map 평균 target 발견 step과 반복 방문을 줄였다.

### Map Runs 영역

각 map별로 다음을 확인한다.

- 모든 map에서 `acc=100%`인지
- 특정 map에서만 좋아진 것이 아니라 여러 unseen map에서 동작하는지
- `target step`이 map별로 얼마나 차이나는지
- `revisit`이 과도한 map이 있는지

## 2. 주장별 필요한 실험

### 주장 1. 제안 방식은 미지 3D 공간에서 target/anomaly cue 기반 탐색이 가능하다

필요 실험:

- 6개 이상 unseen voxel map에서 `success_rate` 확인
- target이 "도착"이 아니라 "RGB-D 관측에 들어온 것"임을 명확히 설명

확인 파일:

```text
results/professor_demo/iceta_semantic_prior/aggregate.csv
results/professor_demo/iceta_semantic_prior/summary.csv
```

### 주장 2A. SSM frontier graph 방식은 MTU3D-style proxy보다 효율적인 탐색 경향을 보였다

필요 실험:

- `mtu3d_proxy` vs `ssm_utility` 비교
- metric: success, avg target step, revisit ratio, observed nodes, frontier switches

현재 사용 가능한 결과:

```text
MTU3D-style Proxy success_rate: 0.917
Proposed SSM success_rate: 1.000
MTU3D-style Proxy avg successful target step: 38.27
Proposed SSM avg successful target step: 22.08
MTU3D-style Proxy revisit: 0.1763
Proposed SSM revisit: 0.0139
```

주의:

- 이 결과는 공식 MTU3D benchmark 재현이 아니라 local simulator proxy 비교이다.
- 따라서 "MTU3D 논문보다 성능이 좋다"가 아니라 "MTU3D-style active frontier-query proxy 대비 본 연구 구조가 현재 simulator에서 더 적은 step/revisit를 보였다"라고 주장한다.

### 주장 2B. 비교논문 MTU3D와 무엇이 다른가

필요 정리:

- dashboard 수치로 MTU3D를 이겼다고 주장하지 않는다.
- MTU3D는 공식 코드/체크포인트/자체 benchmark가 있는 별도 3D-VL exploration 방법이다.
- ICETA에서는 같은 benchmark 재현이 아직 없으므로 "quantitative comparison"이 아니라 "method-level comparison"으로 제시한다.

비교할 축:

| 항목 | MTU3D | 본 연구 |
|---|---|---|
| 주 목표 | 3D visual grounding + embodied navigation | anomaly/target cue 기반 unknown-space exploration |
| 지도/기억 | online query-based 3D spatial memory | explicit online 2D/3D frontier graph |
| frontier 처리 | frontier를 learned query로 표현 | observed-free/unknown boundary를 명시적 node로 생성 |
| 모델 규모 | large 3D vision-language-exploration model | lightweight SSM frontier scorer + graph planner |
| 학습 | 대규모 RGB-D trajectory 기반 VLE pretraining/fine-tuning | synthetic frontier 후보 기반 one-time scorer training |
| 경로 결정 | learned model 중심 | graph update와 SSM scoring/path planning 분리 |
| 배치 방향 | 강한 성능 중심, 경량 onboard가 핵심 검증은 아님 | LIMO ROS2 half-onboard/edge inference를 목표 |

안전한 주장:

> MTU3D는 visual grounding과 active exploration을 통합한 강력한 대형 3D-VL 모델이다. 본 연구는 같은 문제의식 중 unknown-space exploration 부분에 초점을 맞추되, explicit frontier graph와 경량 SSM scorer를 분리해 실제 모바일로봇 배치가 쉬운 구조를 제안한다.

### 주장 3. 모델은 map마다 재학습하지 않고 unseen map에서 동작한다

필요 실험:

- train/test map과 unseen map이 분리되어 있음을 명시
- `models/voxel_frontier_ssm.pt`를 한 번 학습한 뒤, unseen map 평가만 수행

근거:

```text
train frontier rows: 16,910
validation frontier rows: 4,265
best validation top-1: 0.834
unseen maps: 6
```

### 주장 4. 자연어 명령은 경로를 직접 만들지 않고 탐색 조건으로 사용된다

필요 실험:

- command 입력이 `target=red_chair`, `action=log`로 변환되는 것을 보여줌
- dashboard의 `grounding`, `llm_used`, `target`, `action` 표시 확인

주의:

- 현재 결과는 `offline_symbolic_fallback`, `llm_used=false`이다.
- ICETA에서는 "natural-language command grounding prototype" 또는 "symbolic/local grounding"으로 표현한다.
- 대형 LLM/VLM이 경로를 생성한다고 주장하지 않는다.

### 주장 5. 실제 로봇 확장 가능성이 있다

필요 실험/근거:

- 구조적으로 sensor/map/frontier/SSM/Nav2가 분리되어 있음을 설명
- LIMO ROS2에서는 학습이 아니라 onboard inference와 path planning을 수행한다고 설명
- 실제 로봇 구현은 추계전기학술대회로 분리

근거 문서:

```text
docs/limo_ros2_half_onboard_architecture.md
```

## 3. 추가로 하면 좋은 실험

### A. 더 많은 unseen map 평가

목적:

- 6개 map 결과가 우연이 아니라는 점 보강

예시:

```bash
cd /home/mj/my_research/ssm_nav_ws

python3 scripts/run_vln_ssm_voxel_study.py \
  --command "if you find a red chair while exploring the unknown house, leave a log" \
  --out results/iceta_unseen10_simulation \
  --unseen-maps 10 \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2 \
  --llm-provider symbolic
```

### B. mode ablation

목적:

- `mtu3d_proxy`, `ssm_utility`를 비교해 비교논문식 proxy 대비 제안 방식 위치를 명확히 함

예시:

```bash
python3 scripts/run_exploration_research_demo.py \
  --command "if you find a red chair while exploring the unknown house, leave a log" \
  --map-dir maps/iceta_semantic_prior/voxel_unseen \
  --model models/iceta_semantic_prior_ssm.pt \
  --modes mtu3d_proxy ssm_utility \
  --out results/iceta_mode_ablation \
  --limit-maps 12 \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2 \
  --hybrid-ssm-weight 0.35 \
  --ssm-candidate-window 0.12 \
  --ssm-override-margin 0.08 \
  --llm-provider symbolic
```

### C. data efficiency 실험

목적:

- 적은 학습 데이터에서도 frontier scorer가 학습되는지 확인

예시:

```bash
python3 scripts/train_voxel_frontier_ssm.py \
  --train datasets/voxel_frontier_study/train_frontiers.csv \
  --val datasets/voxel_frontier_study/test_frontiers.csv \
  --out models/voxel_frontier_ssm_f025.pt \
  --log results/iceta_data_efficiency/train_f025.csv \
  --epochs 50 \
  --train-fraction 0.25 \
  --seed 710
```

비교할 fraction:

```text
0.10, 0.25, 0.50, 1.00
```

### D. sensing robustness 실험

목적:

- RGB-D 관측 범위/시야가 바뀌어도 탐색이 되는지 확인

변수:

```text
depth_range: 3, 5, 7
aperture: 1, 2, 3
```

### E. latency 최적화 전/후 비교

목적:

- simulator 기준 decision latency와 실제 onboard latency를 분리해 보고, 후속 Jetson/LIMO 환경에서 최적화 목표를 제시

현재 상태:

```text
MTU3D-style Proxy avg_decision_ms: 54.57
Proposed SSM avg_decision_ms: 42.39
```

후속 목표:

```text
TorchScript / ONNX / C++ runtime으로 10~20 ms 이하 목표
```

## 4. ICETA에서 안전하게 주장할 수 있는 문장

사용 가능:

> In twelve unseen semantic-prior voxel maps, the proposed SSM-utility frontier selection achieved 100% target observation success and reduced both successful target observation steps and revisit ratio against the MTU3D-style frontier-query proxy.

사용 가능:

> Compared with MTU3D, our work does not aim to build a larger unified 3D vision-language model. Instead, it keeps spatial memory explicit as an online frontier graph and uses a lightweight SSM-style scorer for deployable exploration planning.

사용 가능:

> The model is trained once on synthetic frontier candidates and reused on unseen maps without per-map retraining.

사용 가능:

> Language is used as a high-level target/anomaly cue, while the actual path inference is performed by the online frontier graph and SSM scorer.

주의해서 사용:

> The current implementation is not a full large-VLM onboard system; the dashboard result uses symbolic command grounding, and real-robot deployment is planned for the follow-up LIMO ROS2 experiment.

피해야 함:

- dashboard 수치로 MTU3D보다 좋다고 말하기.
- MTU3D 공식 benchmark에서 좋다고 말하기.
- ETPNav보다 SR/SPL이 높다.
- 대형 LLM/VLM이 로봇 onboard에서 실시간으로 돈다.
- target까지 물리적으로 도착했다.
