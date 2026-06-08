#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from build_voxel_frontier_dataset import DEFAULT_TARGET_NAMES, normalized_target_names
from local_vln_llm import ground_command
from voxel_frontier_features import FRONTIER_FEATURE_NAMES


WORKSPACE = Path(__file__).resolve().parents[1]


def run_step(args):
    print("$ " + " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], cwd=WORKSPACE, check=True)


def has_maps(path, count):
    return path.exists() and len(list(path.glob("*.vxl"))) >= count


def append_target_args(cmd, target_names):
    out = list(cmd)
    for target_name in target_names:
        out += ["--target-name", target_name]
    return out


def append_target_placement_args(cmd, args):
    out = list(cmd)
    out += ["--placement", args.target_placement]
    if args.replace_targets:
        out.append("--replace-existing")
    return out


def dataset_matches_features(path):
    path = WORKSPACE / path
    if not path.exists():
        return False
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return all(name in header for name in FRONTIER_FEATURE_NAMES) and "target_name" in header


def model_matches_features(path):
    path = WORKSPACE / path
    if not path.exists():
        return False
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return ckpt.get("feature_names") == FRONTIER_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Run local LLM/VLN + one-time SSM frontier + 3D voxel RGB-D study.")
    parser.add_argument("--command", default="전체 집을 순찰하면서 빨간 의자를 발견하면 로그 남겨")
    parser.add_argument("--map-root", default="maps/vln_ssm_voxel_study")
    parser.add_argument("--dataset-root", default="datasets/voxel_frontier_study")
    parser.add_argument("--model", default="models/voxel_frontier_ssm.pt")
    parser.add_argument("--out", default="results/vln_ssm_voxel_study")
    parser.add_argument("--train-maps", type=int, default=40)
    parser.add_argument("--test-maps", type=int, default=10)
    parser.add_argument("--unseen-maps", type=int, default=4)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--depth-range", type=int, default=5)
    parser.add_argument("--aperture", type=int, default=2)
    parser.add_argument("--hybrid-ssm-weight", type=float, default=0.35)
    parser.add_argument("--ssm-candidate-window", type=float, default=0.12)
    parser.add_argument("--ssm-override-margin", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--llm-provider", default="auto", choices=["auto", "ollama", "hf_local", "local_hf", "transformers", "symbolic", "rules", "rule"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-endpoint", default=None)
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--target-name", action="append", default=[], help="Additional target class for multi-target SSM training.")
    parser.add_argument("--target-placement", default="semantic_prior", choices=["random", "semantic_prior"])
    parser.add_argument("--replace-targets", action="store_true")
    parser.add_argument("--regenerate-maps", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    map_root = Path(args.map_root)
    train_dir = map_root / "voxel_train"
    test_dir = map_root / "voxel_test"
    unseen_dir = map_root / "voxel_unseen"
    dataset_root = Path(args.dataset_root)
    train_csv = dataset_root / "train_frontiers.csv"
    test_csv = dataset_root / "test_frontiers.csv"
    model_path = Path(args.model)
    mission = ground_command(args.command, provider="symbolic")
    target_names = normalized_target_names(list(DEFAULT_TARGET_NAMES) + args.target_name + [mission["target_name"]])

    if args.regenerate_maps or not (
        has_maps(WORKSPACE / train_dir, args.train_maps)
        and has_maps(WORKSPACE / test_dir, args.test_maps)
        and has_maps(WORKSPACE / unseen_dir, args.unseen_maps)
    ):
        run_step([
            sys.executable,
            "scripts/generate_voxel_maps.py",
            "--out-root", map_root,
            "--depth", args.depth,
            "--rows", args.rows,
            "--cols", args.cols,
            "--train", args.train_maps,
            "--test", args.test_maps,
            "--unseen", args.unseen_maps,
            "--seed", args.seed,
        ])

    for target_dir in [train_dir, test_dir, unseen_dir]:
        run_step(append_target_placement_args(append_target_args([
            sys.executable,
            "scripts/ensure_semantic_voxel_targets.py",
            "--map-dir", target_dir,
        ], target_names), args))

    if args.rebuild_dataset or not dataset_matches_features(train_csv) or not dataset_matches_features(test_csv):
        run_step(append_target_args([
            sys.executable,
            "scripts/build_voxel_frontier_dataset.py",
            "--map-dir", train_dir,
            "--out", train_csv,
        ], target_names))
        run_step(append_target_args([
            sys.executable,
            "scripts/build_voxel_frontier_dataset.py",
            "--map-dir", test_dir,
            "--out", test_csv,
        ], target_names))

    if args.retrain or not model_matches_features(model_path):
        train_cmd = [
            sys.executable,
            "scripts/train_voxel_frontier_ssm.py",
            "--train", train_csv,
            "--val", test_csv,
            "--out", model_path,
            "--log", Path(args.out) / "train_frontier_ssm_log.csv",
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--train-fraction", args.train_fraction,
            "--seed", args.seed,
        ]
        if args.cpu:
            train_cmd.append("--cpu")
        run_step(train_cmd)
    else:
        print(f"using existing one-time SSM frontier model: {model_path}")

    demo_cmd = [
        sys.executable,
        "scripts/run_exploration_research_demo.py",
        "--command", args.command,
        "--map-dir", unseen_dir,
        "--model", model_path,
        "--modes", "mtu3d_proxy", "ssm_utility",
        "--out", args.out,
        "--limit-maps", args.unseen_maps,
        "--max-steps", args.max_steps,
        "--depth-range", args.depth_range,
        "--aperture", args.aperture,
        "--hybrid-ssm-weight", args.hybrid_ssm_weight,
        "--ssm-candidate-window", args.ssm_candidate_window,
        "--ssm-override-margin", args.ssm_override_margin,
        "--llm-provider", args.llm_provider,
    ]
    if args.llm_model:
        demo_cmd += ["--llm-model", args.llm_model]
    if args.llm_endpoint:
        demo_cmd += ["--llm-endpoint", args.llm_endpoint]
    if args.require_llm:
        demo_cmd.append("--require-llm")
    run_step(demo_cmd)

    run_step([
        sys.executable,
        "scripts/build_voxel_study_dashboard.py",
        "--result-dir", args.out,
        "--map-dir", unseen_dir,
        "--out", Path(args.out) / "dashboard.html",
    ])

    print("")
    print(f"study summary: {Path(args.out) / 'demo_summary.txt'}")
    print(f"aggregate: {Path(args.out) / 'aggregate.csv'}")
    print(f"dashboard: {Path(args.out) / 'dashboard.html'}")


if __name__ == "__main__":
    main()
