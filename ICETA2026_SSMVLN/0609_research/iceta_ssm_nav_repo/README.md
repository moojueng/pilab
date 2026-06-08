# ICETA SSM Frontier Navigation Simulation

This repository package contains the cleaned ICETA simulation code for target-conditioned SSM frontier selection in unknown 3D voxel environments.

## Main Modes

| Mode | Role |
|---|---|
| `mtu3d_proxy` | Local simulator proxy for MTU3D-style active frontier-query exploration |
| `ssm_utility` | Proposed SSM: independent target-conditioned SSM frontier selection over an explicit online frontier graph |
| `hybrid_ssm` | Ablation: MTU3D-style proxy candidate selection plus SSM re-ranking |

## Current Local Simulation Result

| mode | success | avg successful target step | revisit ratio | decision ms |
|---|---:|---:|---:|---:|
| MTU3D-style Proxy | 91.7% | 38.27 | 17.63% | 54.57 |
| Proposed SSM | 100.0% | 22.08 | 1.39% | 42.39 |
| Hybrid SSM | 100.0% | 33.67 | 7.70% | 97.19 |

This is not an official MTU3D benchmark reproduction. The MTU3D-style proxy is a local simulator approximation of the frontier-query exploration idea.

## Reproduce

```bash
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
  --target-placement semantic_prior \
  --replace-targets \
  --regenerate-maps \
  --rebuild-dataset \
  --retrain \
  --llm-provider symbolic
```

## Dashboard

```bash
python3 scripts/voxel_study_dashboard_server.py
```

Open:

```text
http://127.0.0.1:8787
```

## Documentation

See:

```text
docs/iceta_github_upload_step_by_step.md
docs/iceta_2026_simulation_package.md
docs/mtu3d_comparison_note.md
docs/iceta_dashboard_experiment_checklist.md
```

