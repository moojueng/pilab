#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]


def run_step(args):
    print("$ " + " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], cwd=WORKSPACE, check=True)


def has_maps(path, count):
    return path.exists() and len(list(path.glob("*.vxl"))) >= count


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
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--depth-range", type=int, default=5)
    parser.add_argument("--aperture", type=int, default=2)
    parser.add_argument("--hybrid-ssm-weight", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--llm-provider", default="auto", choices=["auto", "ollama", "hf_local", "local_hf", "transformers", "symbolic", "rules", "rule"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-endpoint", default=None)
    parser.add_argument("--require-llm", action="store_true")
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

    run_step([sys.executable, "scripts/ensure_semantic_voxel_targets.py", "--map-dir", unseen_dir])

    if args.rebuild_dataset or not (WORKSPACE / train_csv).exists() or not (WORKSPACE / test_csv).exists():
        run_step([sys.executable, "scripts/build_voxel_frontier_dataset.py", "--map-dir", train_dir, "--out", train_csv])
        run_step([sys.executable, "scripts/build_voxel_frontier_dataset.py", "--map-dir", test_dir, "--out", test_csv])

    if args.retrain or not (WORKSPACE / model_path).exists():
        train_cmd = [
            sys.executable,
            "scripts/train_voxel_frontier_ssm.py",
            "--train", train_csv,
            "--val", test_csv,
            "--out", model_path,
            "--log", Path(args.out) / "train_frontier_ssm_log.csv",
            "--epochs", args.epochs,
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
        "--modes", "utility", "ssm_utility",
        "--out", args.out,
        "--limit-maps", args.unseen_maps,
        "--max-steps", args.max_steps,
        "--depth-range", args.depth_range,
        "--aperture", args.aperture,
        "--hybrid-ssm-weight", args.hybrid_ssm_weight,
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
