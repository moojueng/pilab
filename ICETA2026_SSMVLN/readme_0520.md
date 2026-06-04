# README 0520 - 3D 탐사 경로 계획 / ETPNav 대비 연구 정리

작성일: 2026-05-20
작업 위치: `/home/mj/my_research/ssm_nav_ws`

## 2026-05-26 이어서 정리

오늘 방향은 다음처럼 보강했다.

> 자연어 명령은 local LLM/VLN grounding으로 target/action spec을 만들고, 탐색은 RGB-D online 3D voxel-frontier graph 위에서 수행한다. SSM은 학습 없는 모듈이 아니라, synthetic frontier 후보 데이터로 초기 1회 학습한 경량 frontier scorer이며, unseen map에서는 재학습 없이 inference만 수행한다.

새로 추가/연결한 파일:

```bash
/home/mj/my_research/ssm_nav_ws/scripts/local_vln_llm.py
/home/mj/my_research/ssm_nav_ws/scripts/build_voxel_frontier_dataset.py
/home/mj/my_research/ssm_nav_ws/scripts/train_voxel_frontier_ssm.py
/home/mj/my_research/ssm_nav_ws/scripts/voxel_frontier_features.py
/home/mj/my_research/ssm_nav_ws/scripts/run_vln_ssm_voxel_study.py
```

검증 실행:

```bash
cd /home/mj/my_research/ssm_nav_ws
python3 scripts/run_vln_ssm_voxel_study.py \
  --regenerate-maps \
  --rebuild-dataset \
  --retrain \
  --epochs 35 \
  --cpu \
  --llm-provider symbolic
```

현재는 local HuggingFace LLM grounding server를 사용한다. 기본 모델은 `Qwen/Qwen2.5-1.5B-Instruct`이며, dashboard server가 `hf_local` provider로 호출한다. `mission.json`에서 `llm_used=true`, `grounding_type=local_hf_llm:Qwen/Qwen2.5-1.5B-Instruct`로 기록되어야 진짜 local LLM을 사용한 결과다.

현재 4개 random unseen voxel map 결과:

```text
utility      success_rate=1.0, avg_steps_success=17.25, avg_revisit_ratio=0.0435
ssm_utility  success_rate=1.0, avg_steps_success=16.25, avg_revisit_ratio=0.0000
```

SSM frontier scorer 학습 결과:

```text
train frontier rows=16910, validation rows=4265
best validation top-1 frontier choice accuracy=0.834
```

시뮬레이션 확인용 통합 dashboard:

```bash
/home/mj/my_research/ssm_nav_ws/results/vln_ssm_voxel_study/dashboard.html
```

이 dashboard는 4개 unseen map, `utility`/`ssm_utility` 비교, mission grounding, 학습 top-1, 3D voxel trajectory, target sight line을 한 화면에서 확인한다.

자연어 명령을 dashboard에서 직접 넣어 재실행하려면 정적 HTML을 바로 열지 말고 dashboard server를 띄운다. 이 서버는 local LLM grounding server를 자동으로 확인하고, 없으면 `Qwen/Qwen2.5-1.5B-Instruct`를 로드한다.

```bash
cd /home/mj/my_research/ssm_nav_ws
python3 scripts/voxel_study_dashboard_server.py \
  --host 0.0.0.0 \
  --port 8787 \
  --llm-provider hf_local \
  --local-llm-model Qwen/Qwen2.5-1.5B-Instruct
```

브라우저:

```text
http://127.0.0.1:8787
```

대시보드의 `Natural Language Command` 입력창에 예를 들어 다음처럼 입력하고 `Run Command`를 누른다.

```text
전체 집을 순찰하면서 파란 침대를 발견하면 로그 남겨
```

그러면 local LLM이 자연어 명령을 `target/action` spec으로 grounding하고, 기존 SSM model과 4개 unseen voxel map을 재사용해 `utility`/`ssm_utility` 평가를 다시 돌리며, `mission.json`, `aggregate.csv`, `dashboard.html`을 갱신한다.

## 1. 오늘 정리된 연구 방향

내 연구 방향은 다음처럼 정리한다.

> ETPNav식 online graph navigation 개념을 목표 도달 중심이 아니라 미지공간 탐색 중심으로 재구성하고, RGB-D 기반 online voxel-frontier graph + 경량 local VLN/object-goal parser + SSM-style policy를 결합해 온보드 친화적인 3D 탐색 시스템을 만드는 것.

핵심 표현:

- ETPNav: 자연어 목표까지 이동하는 goal-directed online topological VLN
- 내 연구: 미지공간을 넓히며 목표가 보이면 인지하는 exploration-first voxel-frontier navigation
- 둘 다 사전 full map 없이 online graph/map을 만들지만, 지도 표현과 목적이 다르다.
- baseline은 VLN benchmark 기반 pre-training/fine-tuning 의존이 크고 Transformer planner가 무겁다.
- 내 연구는 경량 local VLN/object-goal parser와 SSM-style policy를 사용해 데이터/모델/연산량을 줄이는 방향이다.

## 2. Baseline과 내 연구 차이

| 항목 | ETPNav baseline | 내 연구 |
| --- | --- | --- |
| 목적 | 자연어 instruction goal까지 이동 | 미지공간 탐색 중 목표 인지 |
| 지도 | online topological map | online voxel-frontier graph |
| node 의미 | viewpoint, waypoint, visited node, ghost node | 관측된 free voxel, obstacle voxel, frontier 후보 |
| 다음 이동 기준 | instruction과 관련 있어 보이는 node/waypoint | 미관측 공간을 많이 열 수 있는 frontier |
| 모델 | Transformer 기반 VLN/cross-modal planner | 경량 local VLN/object-goal parser + SSM-style policy |
| 학습 의존 | VLN benchmark pre-training/fine-tuning 의존 큼 | 새 환경마다 fine-tuning하지 않고 online graph + frontier/SSM policy로 탐색하는 방향 |
| 온보드성 | Jetson급 소형 장비에는 무거울 가능성 큼 | Jetson Orin급 온보드 실시간 구동 목표 |

쉬운 설명:

- ETPNav topo map은 "목표가 있을 법한 장소 node를 고르는 지도"다.
- 내 voxel-frontier graph는 "아직 안 본 공간을 더 넓히기 위한 탐색 후보 지도"다.
- local VLN은 경로계획 전체를 대신하지 않는다.
- local VLN/object-goal parser는 "무엇을 찾아야 하는지"와 "지금 본 것이 목표인지"를 판단한다.
- 실제 어디로 갈지는 frontier graph와 SSM-style policy가 결정한다.

## 3. 오늘 만든 정리 문서

비교 설명 전문:

```bash
/home/mj/my_research/baseline_vs_my_research_summary.txt
```

교수님 데모용 문서:

```bash
/home/mj/my_research/ssm_nav_ws/docs/professor_demo_brief_2026-05-21.md
```

## 4. 오늘 구현한 주요 변경

### 4.1 3D viewer 개선

파일:

```bash
/home/mj/my_research/ssm_nav_ws/scripts/visualize_voxel_run.py
```

변경 내용:

- `Target sight line` 레이어 추가
- `target_seen_step` 이후 로봇 위치에서 목표 voxel까지 주황색 점선 표시
- 이제 목표 좌표에 도착한 것이 아니라 RGB-D 관측 시야 안에 목표가 들어온 것임을 시각적으로 설명 가능

### 4.2 RGB-D 평가 스크립트에 SSM mode 추가

파일:

```bash
/home/mj/my_research/ssm_nav_ws/scripts/eval_rgbd_voxel_nav.py
```

추가 모드:

- `nearest`
- `utility`
- `ssm`
- `ssm_utility`

중요:

- `ssm_utility`는 SSM-style policy action을 먼저 시도한다.
- action이 unsafe하거나 재방문 위험이 있으면 frontier utility fallback을 사용한다.
- 따라서 교수님께는 "SSM 단독 우위"가 아니라 "SSM policy + frontier fallback hybrid demo"라고 말하는 것이 안전하다.

### 4.3 통합 연구 데모 스크립트 추가

파일:

```bash
/home/mj/my_research/ssm_nav_ws/scripts/run_exploration_research_demo.py
```

이 스크립트가 하는 일:

1. 자연어 명령을 local lightweight parser로 해석
2. 목표 spec 생성: 예시 `red_chair`, action=`log`
3. RGB-D ray observation으로 online observed voxel graph 생성
4. frontier 후보 생성
5. `utility`와 `ssm_utility` 정책 비교 실행
6. metrics, aggregate, demo_summary, 3D viewer 생성

## 5. 오늘 실행한 통합 데모

실행 명령:

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

생성 결과:

```bash
results/professor_demo/lightweight_vln_ssm_frontier/demo_summary.txt
results/professor_demo/lightweight_vln_ssm_frontier/aggregate.csv
results/professor_demo/lightweight_vln_ssm_frontier/mission.json
results/professor_demo/lightweight_vln_ssm_frontier/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html
```

현재 aggregate 결과:

```text
utility      success_rate=1.0, avg_steps_success=20.83, avg_revisit_ratio=0.0079
ssm_utility  success_rate=1.0, avg_steps_success=21.67, avg_revisit_ratio=0.0100
```

주의:

- 성공 기준은 목표 좌표 도착이 아니라 `target_seen_step`, 즉 목표가 RGB-D observation 안에 들어온 순간이다.
- 이 데모는 "제안 구조가 동작함을 보이는 proof-of-concept"다.
- 실제 local VLM/VLN object verifier와 Jetson Orin latency 측정은 아직 다음 단계다.

## 6. 쉬고 돌아와서 바로 확인할 명령

```bash
cd /home/mj/my_research/ssm_nav_ws
cat results/professor_demo/lightweight_vln_ssm_frontier/demo_summary.txt
cat results/professor_demo/lightweight_vln_ssm_frontier/aggregate.csv
xdg-open results/professor_demo/lightweight_vln_ssm_frontier/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html
```

viewer에서 확인할 것:

- `Full-map obstacles` 켜기
- `Target sight line` 켜기
- 마지막 step으로 이동
- 주황색 점선이 "목표가 시야에 들어온 순간"을 의미함

## 7. 교수님께 말할 안전한 문장

> 현재는 완성형 local VLM/VLN 전체 시스템은 아니지만, 자연어 목표를 local lightweight parser로 해석하고, 사전 지도 없이 RGB-D 관측으로 online voxel-frontier graph를 만들며, utility/SSM hybrid policy로 미지공간을 탐색해 목표가 시야에 들어오는 것을 6개 unseen map에서 검증했습니다.

더 짧게:

> ETPNav는 목표 도달 중심의 online topological VLN이고, 제 연구는 미지공간 탐색 중심의 lightweight voxel-frontier exploration입니다.

## 8. 지금 미완성/다음 할 일

다음에 이어서 할 일:

1. Gazebo 결과를 새로 돌려서 `results/gazebo_rgbd` warning 제거
2. 실제 camera stream 기반 local object/VLN verifier 연결
3. `ssm_utility` fallback 비율 줄이기
4. SSM 단독/utility/ssm_utility 비교를 더 공정하게 정리
5. Jetson Orin 기준 inference time, memory, FPS 측정 항목 추가
6. 발표용 그림 2개 만들기
   - ETPNav topo map vs 내 voxel-frontier graph
   - local VLN parser + RGB-D mapper + SSM policy 구조도

## 9. 현재 주의할 점

- "ETPNav보다 성능이 좋다"라고 말하지 않는다.
- ETPNav와 현재 결과는 task/metric/dataset이 달라 직접 비교 불가다.
- "LLM을 완전히 붙였다"라고 말하지 않는다. 현재는 local lightweight rule parser이다.
- "SSM만으로 다 해결했다"라고 말하지 않는다. 현재 데모는 SSM policy + frontier utility fallback이다.
- "목표에 도착했다"라고 말하지 않는다. 현재 성공은 목표가 RGB-D 시야에 들어온 것이다.

## 10. 오늘 마지막 상태

검증 완료:

```bash
python3 -m py_compile scripts/eval_rgbd_voxel_nav.py scripts/run_exploration_research_demo.py scripts/visualize_voxel_run.py scripts/check_demo_outputs.py
```

통합 데모 실행 완료:

```bash
python3 scripts/run_exploration_research_demo.py ...
```

남은 warning:

- `scripts/check_demo_outputs.py`에서 Gazebo 산출물이 오래되어 warning이 남아 있음
- Python 연구 데모는 정상
- Gazebo는 발표 전 한 번 새로 돌려 최신 `metrics.csv`, `observed_voxels.csv`, `target_events.csv`를 만드는 것이 좋음
