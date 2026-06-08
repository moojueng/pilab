#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]


class ServerConfig:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.result_dir = Path(args.result_dir)
        self.map_root = Path(args.map_root)
        self.dataset_root = Path(args.dataset_root)
        self.model = Path(args.model)
        self.llm_provider = args.llm_provider
        self.llm_model = args.llm_model
        self.llm_endpoint = args.llm_endpoint
        self.local_llm_model = args.local_llm_model
        self.local_llm_host = args.local_llm_host
        self.local_llm_port = args.local_llm_port
        self.local_llm_cpu = args.local_llm_cpu
        self.no_auto_start_llm = args.no_auto_start_llm
        self.unseen_maps = args.unseen_maps
        self.max_steps = args.max_steps
        self.depth_range = args.depth_range
        self.aperture = args.aperture
        self.hybrid_ssm_weight = args.hybrid_ssm_weight
        self.ssm_candidate_window = args.ssm_candidate_window
        self.ssm_override_margin = args.ssm_override_margin
        self.python_executable = args.python_executable


def default_python_executable():
    conda_python = Path("/home/mj/miniconda3/bin/python3")
    if conda_python.exists():
        return str(conda_python)
    return sys.executable


CONFIG = None
LLM_PROCESS = None


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def local_llm_url(path="/health"):
    return f"http://{CONFIG.local_llm_host}:{CONFIG.local_llm_port}{path}"


def local_llm_health(timeout=2.0):
    try:
        with urllib.request.urlopen(local_llm_url("/health"), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def ensure_local_llm_ready():
    global LLM_PROCESS
    if CONFIG.llm_provider not in ("hf_local", "local_hf", "transformers"):
        return
    if local_llm_health() is not None:
        return
    if CONFIG.no_auto_start_llm:
        raise RuntimeError(
            "local LLM server is not running. Start scripts/local_llm_grounding_server.py or remove --no-auto-start-llm."
        )
    args = [
        CONFIG.python_executable,
        "scripts/local_llm_grounding_server.py",
        "--host",
        CONFIG.local_llm_host,
        "--port",
        str(CONFIG.local_llm_port),
        "--model",
        CONFIG.local_llm_model,
    ]
    if CONFIG.local_llm_cpu:
        args.append("--cpu")
    log_path = Path("/tmp/s_nav_local_llm_grounding_server.log")
    log_file = open(log_path, "a")
    LLM_PROCESS = subprocess.Popen(
        args,
        cwd=WORKSPACE,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if LLM_PROCESS.poll() is not None:
            raise RuntimeError(f"local LLM server exited early. See {log_path}")
        health = local_llm_health(timeout=1.5)
        if health is not None:
            return
        time.sleep(1.0)
    raise RuntimeError("local LLM server did not become ready within 180 seconds")


def run_command(command):
    ensure_local_llm_ready()
    args = [
        CONFIG.python_executable,
        "scripts/run_vln_ssm_voxel_study.py",
        "--command",
        command,
        "--map-root",
        str(CONFIG.map_root),
        "--dataset-root",
        str(CONFIG.dataset_root),
        "--model",
        str(CONFIG.model),
        "--out",
        str(CONFIG.result_dir),
        "--unseen-maps",
        str(CONFIG.unseen_maps),
        "--max-steps",
        str(CONFIG.max_steps),
        "--depth-range",
        str(CONFIG.depth_range),
        "--aperture",
        str(CONFIG.aperture),
        "--hybrid-ssm-weight",
        str(CONFIG.hybrid_ssm_weight),
        "--ssm-candidate-window",
        str(CONFIG.ssm_candidate_window),
        "--ssm-override-margin",
        str(CONFIG.ssm_override_margin),
        "--llm-provider",
        CONFIG.llm_provider,
    ]
    if CONFIG.llm_model:
        args += ["--llm-model", CONFIG.llm_model]
    if CONFIG.llm_endpoint:
        args += ["--llm-endpoint", CONFIG.llm_endpoint]
    elif CONFIG.llm_provider in ("hf_local", "local_hf", "transformers"):
        args += ["--llm-endpoint", local_llm_url("/api/ground")]
    if CONFIG.llm_provider in ("hf_local", "local_hf", "transformers", "ollama"):
        args.append("--require-llm")

    result = subprocess.run(
        args,
        cwd=WORKSPACE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=420,
        check=False,
    )
    aggregate = read_csv_rows(WORKSPACE / CONFIG.result_dir / "aggregate.csv")
    mission_path = WORKSPACE / CONFIG.result_dir / "mission.json"
    mission = json.loads(mission_path.read_text()) if mission_path.exists() else {}
    return {
        "ok": result.returncode == 0,
        "code": result.returncode,
        "message": "completed" if result.returncode == 0 else "command run failed",
        "aggregate": aggregate,
        "mission": mission,
        "output_tail": result.stdout[-5000:],
    }


def ensure_dashboard():
    dashboard = WORKSPACE / CONFIG.result_dir / "dashboard.html"
    if dashboard.exists():
        return dashboard
    subprocess.run(
        [
            CONFIG.python_executable,
            "scripts/build_voxel_study_dashboard.py",
            "--result-dir",
            str(CONFIG.result_dir),
            "--map-dir",
            str(CONFIG.map_root / "voxel_unseen"),
            "--out",
            str(dashboard),
        ],
        cwd=WORKSPACE,
        check=True,
    )
    return dashboard


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/dashboard.html"):
            try:
                path = ensure_dashboard()
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_json({"ok": False, "message": str(exc)}, status=500)
            return
        if self.path == "/api/status":
            self.send_json({
                "ok": True,
                "aggregate": read_csv_rows(WORKSPACE / CONFIG.result_dir / "aggregate.csv"),
            })
            return
        self.send_json({"ok": False, "message": "not found"}, status=404)

    def do_POST(self):
        if self.path == "/api/run_command":
            try:
                payload = read_json_body(self)
                command = str(payload.get("command", "")).strip()
                if not command:
                    self.send_json({"ok": False, "message": "empty command"}, status=400)
                    return
                self.send_json(run_command(command))
            except subprocess.TimeoutExpired:
                self.send_json({"ok": False, "message": "command timed out"}, status=500)
            except Exception as exc:
                self.send_json({"ok": False, "message": str(exc)}, status=500)
            return
        self.send_json({"ok": False, "message": "not found"}, status=404)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(description="Serve the 3D voxel study dashboard with a natural-language command runner.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--result-dir", default="results/professor_demo/iceta_semantic_prior")
    parser.add_argument("--map-root", default="maps/iceta_semantic_prior")
    parser.add_argument("--dataset-root", default="datasets/iceta_semantic_prior")
    parser.add_argument("--model", default="models/iceta_semantic_prior_ssm.pt")
    parser.add_argument("--llm-provider", default="symbolic", choices=["auto", "ollama", "hf_local", "local_hf", "transformers", "symbolic", "rules", "rule"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-endpoint", default=None)
    parser.add_argument("--local-llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--local-llm-host", default="127.0.0.1")
    parser.add_argument("--local-llm-port", type=int, default=8790)
    parser.add_argument("--local-llm-cpu", action="store_true")
    parser.add_argument("--no-auto-start-llm", action="store_true")
    parser.add_argument("--unseen-maps", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--depth-range", type=int, default=5)
    parser.add_argument("--aperture", type=int, default=2)
    parser.add_argument("--hybrid-ssm-weight", type=float, default=0.35)
    parser.add_argument("--ssm-candidate-window", type=float, default=0.12)
    parser.add_argument("--ssm-override-margin", type=float, default=0.08)
    parser.add_argument("--python-executable", default=default_python_executable())
    args = parser.parse_args()
    CONFIG = ServerConfig(args)

    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), Handler)
    print(f"Voxel study dashboard server: http://{CONFIG.host}:{CONFIG.port}")
    print(f"Workspace: {WORKSPACE}")
    print("Enter a natural-language command in the dashboard and press Run Command.")
    server.serve_forever()


if __name__ == "__main__":
    main()
