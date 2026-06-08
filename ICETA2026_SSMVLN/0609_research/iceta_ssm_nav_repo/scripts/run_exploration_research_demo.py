#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from eval_rgbd_voxel_nav import RgbdVoxelExplorer, load_ssm_policy
from ensure_semantic_voxel_targets import ensure_dir
from local_vln_llm import ground_command
from visualize_voxel_run import build_html
from voxel_nav_common import load_voxel_grid


COLOR_KEYWORDS = {
    "red": ["red", "빨간", "빨강", "붉은"],
    "blue": ["blue", "파란", "파랑"],
    "green": ["green", "초록", "녹색"],
    "yellow": ["yellow", "노란", "노랑"],
}

OBJECT_KEYWORDS = {
    "chair": ["chair", "의자"],
    "bed": ["bed", "침대"],
    "table": ["table", "책상", "테이블"],
    "target": ["target", "목표"],
}

ACTION_KEYWORDS = {
    "log": ["log", "record", "기록", "로그"],
    "patrol": ["patrol", "순찰"],
    "explore": ["explore", "탐색"],
}


def first_match(text, table, default):
    lower = text.lower()
    for key, words in table.items():
        if any(word in lower for word in words):
            return key
    return default


def parse_command(command):
    return ground_command(command, provider="symbolic")


def aggregate(rows, modes):
    def avg_metric(subset, name):
        return sum(float(r.get(name, 0.0)) for r in subset) / max(len(subset), 1)

    def max_metric(subset, name):
        return max((float(r.get(name, 0.0)) for r in subset), default=0.0)

    aggs = []
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        n = len(subset)
        successful = [r for r in subset if int(r["success"]) == 1]
        aggs.append({
            "mode": mode,
            "maps": n,
            "success_rate": sum(int(r["success"]) for r in subset) / n,
            "avg_steps_all": sum(float(r["steps"]) for r in subset) / n,
            "avg_steps_success": sum(float(r["steps"]) for r in successful) / max(len(successful), 1),
            "avg_revisit_ratio": sum(float(r["revisit_ratio"]) for r in subset) / n,
            "avg_observed_nodes": sum(float(r["observed_nodes"]) for r in subset) / n,
            "avg_frontier_switches": sum(float(r["frontier_switches"]) for r in subset) / n,
            "avg_fallback_count": sum(float(r["fallback_count"]) for r in subset) / n,
            "avg_frontier_count": avg_metric(subset, "avg_frontier_count"),
            "max_frontier_count": max_metric(subset, "max_frontier_count"),
            "avg_step_wall_ms": avg_metric(subset, "avg_step_wall_ms"),
            "avg_decision_wall_ms": avg_metric(subset, "avg_decision_wall_ms"),
            "max_decision_wall_ms": max_metric(subset, "max_decision_wall_ms"),
        })
    return aggs


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, mission, aggs, viewer_path):
    lines = [
        "Local LLM/VLN + RGB-D Voxel-Frontier + SSM Frontier Policy Demo",
        "",
        f"command: {mission['raw_command']}",
        f"command grounding: {mission.get('grounding_type', 'unknown')} "
        f"(llm_used={mission.get('llm_used', False)}, confidence={float(mission.get('confidence', 0.0)):.2f})",
        f"target spec: target={mission['target_name']}, action={mission['on_detection_action']}",
        "",
        "What this demo validates:",
        "- no prebuilt map is loaded by the online explorer",
        "- a natural-language command is grounded into a target/object-goal spec before exploration",
        "- RGB-D ray observation updates an online observed voxel graph",
        "- frontier candidates are selected to expand unknown space",
        "- ssm_utility mode uses a one-time-trained target-conditioned SSM-style scorer over explicit frontier-graph candidates",
        "- hybrid_ssm mode is kept as an ablation that re-ranks MTU3D-style proxy candidates with the SSM scorer",
        "- success means the target entered RGB-D observation, not that the robot physically arrived at the target voxel",
        "",
        "Aggregate:",
    ]
    for row in aggs:
        lines.append(
            f"- {row['mode']}: success_rate={row['success_rate']:.3f}, "
            f"avg_steps_success={row['avg_steps_success']:.2f}, "
            f"avg_revisit_ratio={row['avg_revisit_ratio']:.4f}, "
            f"avg_fallback_count={row['avg_fallback_count']:.2f}, "
            f"avg_decision_ms={row.get('avg_decision_wall_ms', 0.0):.3f}, "
            f"avg_frontiers={row.get('avg_frontier_count', 0.0):.2f}"
        )
    lines += [
        "",
        f"viewer: {viewer_path}",
        "",
        "Honest scope:",
        "- This is a demo validation for the proposed architecture direction.",
        "- A real local VLM/VLN object verifier and Jetson Orin timing measurement are still next-step work.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def run_demo(args):
    mission = ground_command(
        args.command,
        provider=args.llm_provider,
        model=args.llm_model,
        endpoint=args.llm_endpoint,
        require_llm=args.require_llm,
    )
    ensure_dir(args.map_dir, target_names=[mission["target_name"]])
    map_paths = sorted(Path(args.map_dir).glob("*.vxl"))
    if args.limit_maps > 0:
        map_paths = map_paths[:args.limit_maps]
    if not map_paths:
        raise RuntimeError(f"no .vxl maps found in {args.map_dir}")

    modes = args.modes
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "mission.json").write_text(json.dumps(mission, ensure_ascii=False, indent=2) + "\n")

    model = None
    if any(mode in ("ssm", "ssm_utility", "hybrid_ssm") for mode in modes):
        model = load_ssm_policy(args.model)

    rows = []
    for map_path in map_paths:
        grid = load_voxel_grid(map_path)
        map_name = map_path.stem
        for mode in modes:
            explorer = RgbdVoxelExplorer(
                grid,
                mode=mode,
                max_steps=args.max_steps,
                depth_range=args.depth_range,
                aperture=args.aperture,
                model=model,
                target_name=mission["target_name"],
                hybrid_ssm_weight=args.hybrid_ssm_weight,
                ssm_candidate_window=args.ssm_candidate_window,
                ssm_override_margin=args.ssm_override_margin,
            )
            metrics = explorer.run()
            case_out = root / mode / map_name
            explorer.save_outputs(case_out)
            row = {"map": map_name, **metrics.__dict__}
            rows.append(row)
            write_csv(case_out / "metrics.csv", [metrics.__dict__])
            print(row)

    aggs = aggregate(rows, modes)
    write_csv(root / "summary.csv", rows)
    write_csv(root / "aggregate.csv", aggs)

    viewer_mode = "ssm_utility" if "ssm_utility" in modes else modes[0]
    viewer_map = map_paths[0]
    viewer_case = root / viewer_mode / viewer_map.stem
    viewer_path = viewer_case / "rgbd_ssm_frontier_view.html"
    build_html(argparse.Namespace(
        map=str(viewer_map),
        trajectory=str(viewer_case / "trajectory.csv"),
        observed=str(viewer_case / "observed_voxels.csv"),
        metrics=str(viewer_case / "metrics.csv"),
        out=str(viewer_path),
        title=f"{viewer_map.stem} / {viewer_mode} / {mission['target_name']}",
    ))

    write_summary(root / "demo_summary.txt", mission, aggs, viewer_path)
    print(f"\nDemo summary: {root / 'demo_summary.txt'}")
    print(f"Viewer: {viewer_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--command",
        default="if you find a red chair while exploring the unknown house, leave a log",
    )
    p.add_argument("--map-dir", default="maps/voxel_unseen")
    p.add_argument("--model", default="models/voxel_frontier_ssm.pt")
    p.add_argument("--modes", nargs="+", default=["mtu3d_proxy", "ssm_utility"], choices=["nearest", "utility", "mtu3d_proxy", "ssm", "ssm_utility", "hybrid_ssm"])
    p.add_argument("--out", default="results/professor_demo/lightweight_vln_ssm_frontier")
    p.add_argument("--limit-maps", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=180)
    p.add_argument("--depth-range", type=int, default=5)
    p.add_argument("--aperture", type=int, default=2)
    p.add_argument("--hybrid-ssm-weight", type=float, default=0.35)
    p.add_argument("--ssm-candidate-window", type=float, default=0.12)
    p.add_argument("--ssm-override-margin", type=float, default=0.08)
    p.add_argument("--llm-provider", default="auto", choices=["auto", "ollama", "hf_local", "local_hf", "transformers", "symbolic", "rules", "rule"])
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-endpoint", default=None)
    p.add_argument("--require-llm", action="store_true")
    args = p.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
