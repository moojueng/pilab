# ICETA 2026 Simulation GitHub Upload Step-by-Step

작성일: 2026-06-09  
프로젝트 경로: `/home/mj/my_research/ssm_nav_ws`

## 1. 현재 연구 코드 상태 요약

ICETA 학회 제출용 시뮬레이션은 다음 구조로 정리한다.

| 구분 | 코드/모드 | 역할 |
|---|---|---|
| 비교군 | `mtu3d_proxy` | MTU3D의 active frontier-query 탐색 관점을 local voxel simulator에서 근사한 비교군 |
| 제안 방식 | `ssm_utility` / Proposed SSM | MTU3D-style proxy를 호출하지 않는 독립 target-conditioned SSM frontier selection |
| 참고 실험 | `hybrid_ssm` | proxy 후보를 SSM으로 re-ranking하는 ablation |
| VLN/topological reference | ETPNav | 목표지점 기반 VLN/topological navigation 참고 구조. 직접적인 exploration baseline은 아님 |

현재 발표 주장은 다음처럼 제한한다.

> 동일 local semantic-prior voxel simulator에서 독립 Proposed SSM은 MTU3D-style proxy 대비 target observation success, average target step, revisit ratio, prototype decision time을 개선하였다.

주의:

- 공식 MTU3D benchmark를 재현한 결과는 아니다.
- 따라서 “MTU3D 논문 전체를 이겼다”가 아니라 “MTU3D-style proxy를 동일 simulator에서 이겼다”라고 표현한다.
- official MTU3D와의 직접 비교는 공식 코드/체크포인트/동일 benchmark adapter가 필요하다.

## 2. GitHub에 올릴 핵심 코드

필수 업로드 대상:

```text
README.md
.gitignore
scripts/voxel_nav_common.py
scripts/voxel_frontier_features.py
scripts/generate_voxel_maps.py
scripts/ensure_semantic_voxel_targets.py
scripts/build_voxel_frontier_dataset.py
scripts/train_voxel_frontier_ssm.py
scripts/eval_rgbd_voxel_nav.py
scripts/run_exploration_research_demo.py
scripts/run_vln_ssm_voxel_study.py
scripts/build_voxel_study_dashboard.py
scripts/voxel_study_dashboard_server.py
scripts/visualize_voxel_run.py
scripts/local_vln_llm.py
scripts/local_llm_grounding_server.py
docs/mtu3d_comparison_note.md
docs/iceta_2026_simulation_package.md
docs/iceta_dashboard_experiment_checklist.md
docs/limo_ros2_half_onboard_architecture.md
docs/iceta_github_upload_step_by_step.md
src/s_nav_core/
src/s_nav_msgs/
```

선택 업로드 대상:

```text
models/iceta_semantic_prior_ssm.pt
results/professor_demo/iceta_semantic_prior/aggregate.csv
results/professor_demo/iceta_semantic_prior/demo_summary.txt
results/professor_demo/iceta_semantic_prior/dashboard.html
results/professor_demo/iceta_semantic_prior/ssm_utility/unseen_001/rgbd_ssm_frontier_view.html
```

선택 업로드 대상은 exact demo 확인용이다. GitHub repo에는 넣지 않고 GitHub Release artifact로 올리는 방식도 가능하다.

## 3. GitHub에 올리지 말아야 할 것

다음은 로컬 생성물 또는 대용량/중복 산출물이므로 기본적으로 제외한다.

```text
build/
install/
log/
logs/
scripts/__pycache__/
datasets/
maps/
results/
models/
*.pt
*.onnx
*.zip
debug_*.txt
graph_dump.csv
```

이유:

- `build/`, `install/`, `log/`는 ROS2/colcon 로컬 빌드 산출물이다.
- `datasets/`는 재생성 가능하며 현재 `datasets/iceta_semantic_prior`만 약 86MB이다.
- `maps/`, `results/`, `models/`는 재현 명령으로 다시 만들 수 있다.
- 최종 논문용 결과만 필요한 경우 `aggregate.csv`, `demo_summary.txt`, `dashboard.html`만 별도 release artifact로 첨부한다.

## 4. GitHub 업로드 전 코드 정리 순서

1. 프로젝트 루트로 이동한다.

```bash
cd /home/mj/my_research/ssm_nav_ws
```

2. 문법 검사를 실행한다.

```bash
python3 -m py_compile \
  scripts/voxel_nav_common.py \
  scripts/voxel_frontier_features.py \
  scripts/ensure_semantic_voxel_targets.py \
  scripts/build_voxel_frontier_dataset.py \
  scripts/train_voxel_frontier_ssm.py \
  scripts/eval_rgbd_voxel_nav.py \
  scripts/run_exploration_research_demo.py \
  scripts/run_vln_ssm_voxel_study.py \
  scripts/build_voxel_study_dashboard.py \
  scripts/voxel_study_dashboard_server.py \
  scripts/local_vln_llm.py \
  scripts/local_llm_grounding_server.py
```

3. 불필요한 생성물이 Git에 잡히지 않도록 `.gitignore`를 확인한다.

```bash
cat .gitignore
```

4. Git 저장소를 새로 만들 경우 초기화한다.

```bash
git init
git add README.md .gitignore scripts src docs
git status
```

5. 모델/결과 파일을 GitHub repo에 강제로 포함할지 결정한다.

권장 방식:

```text
코드 repo: scripts, src, docs만 업로드
GitHub Release: model/checkpoint와 dashboard 결과 첨부
```

만약 작은 demo model만 repo에 포함하고 싶으면:

```bash
git add -f models/iceta_semantic_prior_ssm.pt
git add -f results/professor_demo/iceta_semantic_prior/aggregate.csv
git add -f results/professor_demo/iceta_semantic_prior/demo_summary.txt
git add -f results/professor_demo/iceta_semantic_prior/dashboard.html
```

6. 첫 commit을 만든다.

```bash
git commit -m "Prepare ICETA simulation artifact"
```

7. GitHub 원격 저장소를 연결한다.

```bash
git remote add origin https://github.com/<USER>/<REPO>.git
git branch -M main
git push -u origin main
```

## 5. ICETA 시뮬레이션 재현 Step-by-Step

### Step 1. 환경 준비

Python 3 환경에서 다음 패키지가 필요하다.

```text
torch
numpy
```

대시보드는 별도 웹 프레임워크 없이 Python HTTP server와 정적 HTML로 동작한다.

### Step 2. 전체 ICETA 실험 재생성

아래 명령은 semantic-prior voxel map 생성, frontier dataset 생성, SSM 학습, 12개 unseen map 평가, dashboard 생성을 한 번에 수행한다.

```bash
cd /home/mj/my_research/ssm_nav_ws

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python3 scripts/run_vln_ssm_voxel_study.py \
  --command "if you find a red chair while exploring the unknown house, leave a log" \
  --map-root maps/iceta_semantic_prior \
  --dataset-root datasets/iceta_semantic_prior \
  --model models/iceta_semantic_prior_ssm.pt \
  --out results/professor_demo/iceta_semantic_prior \
  --train-maps 40 \
  --test-maps 10 \
  --unseen-maps 12 \
  --epochs 30 \
  --batch-size 2048 \
  --max-steps 180 \
  --depth-range 5 \
  --aperture 2 \
  --hybrid-ssm-weight 0.35 \
  --ssm-candidate-window 0.12 \
  --ssm-override-margin 0.08 \
  --target-placement semantic_prior \
  --replace-targets \
  --regenerate-maps \
  --rebuild-dataset \
  --retrain \
  --llm-provider symbolic
```

### Step 3. 빠른 재평가

이미 model/map/dataset이 있다면 재학습 없이 평가만 실행한다.

```bash
cd /home/mj/my_research/ssm_nav_ws

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python3 scripts/run_exploration_research_demo.py \
  --command "if you find a red chair while exploring the unknown house, leave a log" \
  --map-dir maps/iceta_semantic_prior/voxel_unseen \
  --model models/iceta_semantic_prior_ssm.pt \
  --modes mtu3d_proxy ssm_utility hybrid_ssm \
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

### Step 4. 대시보드 실행

```bash
cd /home/mj/my_research/ssm_nav_ws
python3 scripts/voxel_study_dashboard_server.py
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8787
```

### Step 5. 대시보드에서 확인할 것

1. `Mode`에서 `MTU3D-style Proxy` 선택
2. `Mode`에서 `Proposed SSM` 선택
3. `Hybrid SSM`은 참고 ablation으로만 확인
4. `Map`에서 `unseen_007`, `unseen_003`, `unseen_012`를 우선 확인
5. timeline slider를 움직이며 trajectory, observed voxel, target observation step 확인

## 6. 현재 최종 결과

결과 파일:

```text
results/professor_demo/iceta_semantic_prior/aggregate.csv
results/professor_demo/iceta_semantic_prior/demo_summary.txt
results/professor_demo/iceta_semantic_prior/dashboard.html
```

최종 aggregate:

| mode | success | avg successful target step | revisit ratio | decision ms |
|---|---:|---:|---:|---:|
| MTU3D-style Proxy | 91.7% | 38.27 | 17.63% | 54.57 |
| Proposed SSM | 100.0% | 22.08 | 1.39% | 42.39 |
| Hybrid SSM | 100.0% | 33.67 | 7.70% | 97.19 |

해석:

- `Proposed SSM`은 `MTU3D-style Proxy`보다 target 발견 성공률이 높다.
- 평균 target 발견 step이 `38.27 -> 22.08`로 감소했다.
- 재방문율이 `17.63% -> 1.39%`로 감소했다.
- local Python simulator 기준 decision time도 `54.57 ms -> 42.39 ms`로 감소했다.
- `Hybrid SSM`은 proxy+SSM ablation이며 주 연구 방식이 아니다.

## 7. 논문/발표에서 쓸 안전한 표현

권장 문장:

> In a local semantic-prior voxel simulation with 12 unseen maps, the proposed target-conditioned SSM frontier selection achieved 100% target observation success and reduced average successful target discovery steps and revisit ratio compared with an MTU3D-style frontier-query proxy.

한국어 표현:

> 본 연구의 local semantic-prior voxel simulation에서 독립 Proposed SSM은 MTU3D-style frontier-query proxy 대비 target observation success를 높이고, 평균 target discovery step과 revisit ratio를 감소시켰다.

주의 문장:

> This is not an official MTU3D benchmark reproduction. The MTU3D-style proxy is a local simulator approximation of the frontier-query exploration idea.

## 8. ICETA 제출 전 체크리스트

- [ ] `aggregate.csv` 수치 확인
- [ ] dashboard에서 `MTU3D-style Proxy`, `Proposed SSM`, `Hybrid SSM` 구분 확인
- [ ] `Proposed SSM`이 proxy를 호출하지 않는 독립 방식임을 발표자료에 명시
- [ ] `Hybrid SSM`은 ablation이라고 명시
- [ ] ETPNav는 direct exploration baseline이 아니라 VLN/topological reference라고 명시
- [ ] MTU3D official benchmark 재현이 아니라 local proxy 비교라고 명시
- [ ] GitHub repo에는 코드와 문서 중심으로 업로드
- [ ] model/result는 repo 포함 여부를 결정하거나 GitHub Release artifact로 분리

