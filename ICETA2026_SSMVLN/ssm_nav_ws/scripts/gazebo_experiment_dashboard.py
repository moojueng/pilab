#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from local_vln_llm import ground_command


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WORKSPACE / "results/gazebo_rgbd/observed_voxels.csv"
DEFAULT_TRAJECTORY = WORKSPACE / "results/gazebo_rgbd/trajectory.csv"
DEFAULT_METRICS = WORKSPACE / "results/gazebo_rgbd/metrics.csv"
DEFAULT_CAMERA = WORKSPACE / "results/gazebo_rgbd/latest_camera.jpg"
DEFAULT_GRAPH_NODES = WORKSPACE / "results/gazebo_rgbd/runtime_graph_nodes.csv"
DEFAULT_GRAPH_EDGES = WORKSPACE / "results/gazebo_rgbd/runtime_graph_edges.csv"
DEFAULT_FRONTIER_FEATURES = WORKSPACE / "results/gazebo_rgbd/frontier_features.csv"
DEFAULT_TARGET_EVENTS = WORKSPACE / "results/gazebo_rgbd/target_events.csv"
DEFAULT_RGBD_RESULTS = WORKSPACE / "results/voxel_sim/rgbd_frontier"


class ExperimentState:
    def __init__(self):
        self.process = None
        self.log = deque(maxlen=500)
        self.started_at = None
        self.lock = threading.Lock()

    def append_log(self, line):
        with self.lock:
            self.log.append(line.rstrip())

    def is_running(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def start(self, launch_file, launch_args=None):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "already running"

            if "rgbd" in launch_file and not os.environ.get("DISPLAY"):
                message = (
                    "DISPLAY is not set. Gazebo RGB-D camera/depth sensors need rendering. "
                    "Start this dashboard from a MobaXterm/X11 terminal where Gazebo GUI can open."
                )
                self.log.append(message)
                return False, message

            cleanup_stale_experiment_processes()
            time.sleep(0.5)

            for path in [
                DEFAULT_OUTPUT,
                DEFAULT_TRAJECTORY,
                DEFAULT_METRICS,
                DEFAULT_CAMERA,
                DEFAULT_GRAPH_NODES,
                DEFAULT_GRAPH_EDGES,
                DEFAULT_FRONTIER_FEATURES,
                DEFAULT_TARGET_EVENTS,
            ]:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

            launch_args = launch_args or {}
            arg_text = " ".join(f"{key}:={value}" for key, value in launch_args.items())
            command = (
                f"cd {quote(str(WORKSPACE))} && "
                "unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONHOME PYTHONPATH && "
                "export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH && "
                "source /opt/ros/humble/setup.bash && "
                "source install/setup.bash && "
                "export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH && "
                "export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://127.0.0.1:11346} && "
                f"ros2 launch s_nav_core {launch_file} {arg_text}"
            )
            self.process = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=WORKSPACE,
                preexec_fn=os.setsid,
                bufsize=1,
            )
            self.started_at = time.time()
            self.log.clear()
            self.log.append(f"$ {command}")

            thread = threading.Thread(target=self._read_log, daemon=True)
            thread.start()
            return True, "started"

    def _read_log(self):
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.append_log(line)
            if "Address already in use" in line or "process has died" in line and "gazebo" in line:
                self.append_log("Gazebo failed to start. Stopping launch process group.")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    pass

    def stop(self):
        with self.lock:
            proc = self.process
            if proc is None or proc.poll() is not None:
                return False, "not running"
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            self.log.append("sent SIGINT to launch process group")
            return True, "stopping"

    def snapshot_log(self):
        with self.lock:
            return list(self.log)

    def uptime(self):
        with self.lock:
            if self.started_at is None or self.process is None or self.process.poll() is not None:
                return 0.0
            return time.time() - self.started_at


STATE = ExperimentState()


def quote(value):
    return "'" + value.replace("'", "'\\''") + "'"


def cleanup_stale_experiment_processes():
    patterns = [
        "ros2 launch s_nav_core",
        "rgbd_frontier_navigator",
        "depth_voxel_mapper",
        "robot_state_publisher",
        "/spawn_entity.py",
        "gzserver",
        "gzclient",
        "gazebo --verbose",
    ]
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    subprocess.run(
        ["fuser", "-k", "11345/tcp", "11346/tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_ros_command(command, timeout=4):
    full = (
        f"cd {quote(str(WORKSPACE))} && "
        "unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONHOME PYTHONPATH && "
        "export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH && "
        "source /opt/ros/humble/setup.bash && "
        "source install/setup.bash && "
        "export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH && "
        f"{command}"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", full],
            cwd=WORKSPACE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def topic_list():
    code, output = run_ros_command("ros2 topic list", timeout=3)
    if code != 0:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def publish_cmd_vel(linear, angular):
    command = (
        "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "
        f"'{{linear: {{x: {linear}, y: 0.0, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: {angular}}}}}'"
    )
    return run_ros_command(command, timeout=4)


def infer_goal_from_command(text):
    raw = (text or "").strip()
    try:
        grounded = ground_command(
            raw,
            provider=os.environ.get("S_NAV_LLM_PROVIDER", "auto"),
            model=os.environ.get("S_NAV_LLM_MODEL"),
            endpoint=os.environ.get("S_NAV_LLM_ENDPOINT"),
            require_llm=os.environ.get("S_NAV_REQUIRE_LLM", "0") == "1",
        )
    except Exception as exc:
        return {
            "ok": False,
            "goal": None,
            "label": None,
            "reason": str(exc),
            "command": raw,
        }

    target_object = grounded.get("target_object")
    target_color = grounded.get("target_color")
    if target_object == "bed":
        return {
            "ok": True,
            "goal": "bed",
            "label": "파란 침대" if target_color == "blue" else f"{target_color} bed",
            "mission_mode": grounded.get("mission_mode", "coverage_patrol"),
            "reason": grounded.get("reason", ""),
            "command": raw,
            "grounding": grounded,
        }
    if target_object == "chair":
        return {
            "ok": True,
            "goal": "chair",
            "label": "빨간 의자" if target_color == "red" else f"{target_color} chair",
            "mission_mode": grounded.get("mission_mode", "coverage_patrol"),
            "reason": grounded.get("reason", ""),
            "command": raw,
            "grounding": grounded,
        }
    return {
        "ok": False,
        "goal": None,
        "label": None,
        "reason": "현재 Gazebo demo는 chair 또는 bed 목표만 지원합니다",
        "command": raw,
        "grounding": grounded,
    }


def observed_voxel_summary(path=DEFAULT_OUTPUT):
    path = Path(path)
    summary = {
        "exists": path.exists(),
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "free": 0,
        "occupied": 0,
        "total": 0,
        "mtime": path.stat().st_mtime if path.exists() else None,
    }
    if not path.exists():
        return summary
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = int(row.get("value", -1))
                if value == 0:
                    summary["free"] += 1
                elif value == 1:
                    summary["occupied"] += 1
                summary["total"] += 1
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def exploration_map(voxel_path=DEFAULT_OUTPUT, trajectory_path=DEFAULT_TRAJECTORY):
    grid = {
        "min_x": -4.0,
        "max_x": 7.0,
        "min_y": -4.5,
        "max_y": 4.5,
        "voxel_size": 0.20,
        "cols": 55,
        "rows": 45,
        "free": [],
        "occupied": [],
        "path": [],
        "robot": None,
    }

    voxel_path = Path(voxel_path)
    if voxel_path.exists():
        free = set()
        occupied = set()
        try:
            with open(voxel_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ix = int(row.get("ix", 0))
                    iy = int(row.get("iy", 0))
                    iz = int(row.get("iz", 0))
                    value = int(row.get("value", -1))
                    if iz > 3:
                        continue
                    cell = (ix, iy)
                    if value == 1 and iz >= 2:
                        occupied.add(cell)
                    elif value == 0 and cell not in occupied:
                        free.add(cell)
            free -= occupied
            grid["free"] = [[ix, iy] for ix, iy in sorted(free)]
            grid["occupied"] = [[ix, iy] for ix, iy in sorted(occupied)]
        except Exception as exc:
            grid["error"] = str(exc)

    trajectory_path = Path(trajectory_path)
    if trajectory_path.exists():
        try:
            points = []
            with open(trajectory_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = float(row.get("x", 0.0))
                    y = float(row.get("y", 0.0))
                    yaw = float(row.get("yaw", 0.0))
                    points.append([x, y, yaw])
            grid["path"] = points[-500:]
            if points:
                grid["robot"] = points[-1]
        except Exception as exc:
            grid["trajectory_error"] = str(exc)

    return grid


def read_csv_rows(path, max_rows=60):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path), "rows": [], "fields": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
        return {
            "exists": True,
            "path": str(path),
            "fields": reader.fieldnames or [],
            "rows": rows,
        }


def list_viewers():
    viewers = []
    for root in [DEFAULT_RGBD_RESULTS, WORKSPACE / "results/voxel_sim/frontier_modes_fraction_025"]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.html")):
            viewers.append({
                "name": str(path.relative_to(WORKSPACE)),
                "path": str(path),
            })
    return viewers


def aggregate_files():
    files = []
    for path in sorted((WORKSPACE / "results").rglob("aggregate.csv")):
        files.append({
            "name": str(path.relative_to(WORKSPACE)),
            "path": str(path),
            "data": read_csv_rows(path, max_rows=20),
        })
    return files


def clean_log_line(lines):
    interesting = [
        "Target visible",
        "Target reached",
        "candidate visible",
        "Frontier nav",
        "RGB-D frontier navigator started",
        "Depth voxel mapper started",
        "Depth frame mapped",
        "Successfully spawned",
        "Waiting for odom",
        "Waiting for camera_info/odom",
        "Waiting for observed free cells",
        "No frontier candidates",
        "process has died",
        "Address already in use",
    ]
    for line in reversed(lines):
        if any(key in line for key in interesting):
            return line
    return lines[-1] if lines else ""


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gazebo RGB-D 탐색 대시보드</title>
  <style>
    :root {
      --bg: #f5f2ec;
      --panel: #fffefa;
      --ink: #22272c;
      --muted: #6e747a;
      --line: #d8d0c3;
      --green: #148255;
      --red: #c4362e;
      --blue: #1769aa;
      --orange: #df7138;
      --black: #1d2125;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,254,250,.92);
      position: sticky;
      top: 0;
      z-index: 5;
      backdrop-filter: blur(12px);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .subtitle {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      font-size: 13px;
      font-weight: 700;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--red);
    }
    .dot.on { background: var(--green); }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: calc(100vh - 72px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
    }
    main {
      padding: 18px;
      min-width: 0;
    }
    .section {
      border-top: 1px solid var(--line);
      padding: 16px 0;
    }
    .section:first-child { border-top: 0; padding-top: 0; }
    .section-title {
      margin: 0 0 12px;
      color: #4f565d;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .button-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button, a.button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      min-height: 38px;
      padding: 0 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 750;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
    }
    button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
    button.danger { background: var(--red); border-color: var(--red); color: #fff; }
    button.accent { background: var(--orange); border-color: var(--orange); color: #fff; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-height: 76px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      line-height: 1;
      margin-bottom: 8px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(320px, .7fr);
      gap: 14px;
    }
    .camera-wrap {
      background: #161a1e;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .camera-wrap img {
      width: 100%;
      height: 100%;
      max-height: 620px;
      object-fit: contain;
      display: block;
    }
    .map-wrap {
      background: #272b2f;
      min-height: 420px;
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    #explorationMap {
      width: 100%;
      max-height: 620px;
      aspect-ratio: 55 / 45;
      border: 1px solid #3f464d;
      background: #2f3439;
      display: block;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 2px;
      border: 1px solid rgba(0,0,0,.18);
    }
    .simple-log {
      display: grid;
      gap: 10px;
    }
    .simple-log div {
      border: 1px solid #e1d8ca;
      border-radius: 8px;
      background: #fcfaf5;
      padding: 10px 11px;
      font-size: 14px;
    }
    .hidden-debug { display: none !important; }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .card h2 {
      margin: 0;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
    }
    .card-body { padding: 14px; }
    pre {
      margin: 0;
      min-height: 280px;
      max-height: 420px;
      overflow: auto;
      padding: 12px;
      border-radius: 8px;
      background: #1d2125;
      color: #e8edf2;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid #e6dfd4;
      padding: 7px 8px;
      text-align: left;
      white-space: nowrap;
    }
    th { color: #59616a; font-weight: 800; background: #faf7f0; }
    .topic-list, .viewer-list {
      display: grid;
      gap: 7px;
      max-height: 260px;
      overflow: auto;
    }
    .topic, .viewer {
      padding: 8px 9px;
      border: 1px solid #e3dbce;
      border-radius: 8px;
      background: #fcfaf5;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .cmd-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .cmd-grid .wide { grid-column: span 3; }
    .command-box {
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }
    .command-box textarea {
      width: 100%;
      min-height: 70px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      line-height: 1.35;
    }
    .command-box .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .path {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      margin-top: 8px;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Gazebo RGB-D 탐색 대시보드</h1>
      <div class="subtitle">작은 집 환경에서 RGB-D voxel mapping, frontier 탐색, 목표 발견 상태를 확인합니다</div>
    </div>
    <div class="status-pill"><span id="runDot" class="dot"></span><span id="runText">checking</span></div>
  </header>
  <div class="layout">
    <aside>
      <div class="section">
        <p class="section-title">실험 실행</p>
        <div class="command-box">
          <textarea id="commandInput" placeholder="예: 전체 집을 순찰하면서 빨간 의자를 발견하면 로그 남겨"></textarea>
          <button id="llmStartBtn" class="primary">자연어 명령 해석 후 시작</button>
          <div id="llmHint" class="hint">위 문장에서 목표 대상을 해석해 coverage patrol을 시작합니다.</div>
        </div>
        <select id="goalSelect" style="width:100%; height:38px; margin-bottom:8px; border:1px solid var(--line); border-radius:8px; padding:0 10px; background:#fff;">
          <option value="chair">빨간 의자 찾기</option>
          <option value="bed">파란 침대 찾기</option>
        </select>
        <div class="button-grid">
          <button id="startNavBtn" class="accent" title="드롭다운에서 선택한 목표로 coverage patrol을 시작합니다">선택 목표로 시작</button>
          <button id="stopBtn" class="danger">정지</button>
        </div>
      </div>
      <div class="section hidden-debug">
        <p class="section-title">수동 이동</p>
        <div class="cmd-grid">
          <button data-cmd="left">좌회전</button>
          <button data-cmd="forward" class="accent">전진</button>
          <button data-cmd="right">우회전</button>
          <button data-cmd="spin_left">제자리 좌</button>
          <button data-cmd="stop" class="danger">정지</button>
          <button data-cmd="spin_right">제자리 우</button>
          <button data-cmd="back" class="wide">후진</button>
        </div>
      </div>
      <div class="section">
        <p class="section-title">필수 ROS 토픽</p>
        <div id="topicList" class="topic-list"></div>
      </div>
      <div class="section hidden-debug">
        <p class="section-title">3D 결과 뷰어</p>
        <div id="viewerList" class="viewer-list"></div>
      </div>
    </aside>
    <main>
      <div class="metrics">
        <div class="metric"><strong id="mObserved">-</strong><span>관측 voxel</span></div>
        <div class="metric"><strong id="mOccupied">-</strong><span>장애물 voxel</span></div>
        <div class="metric"><strong id="mUptime">-</strong><span>실행 시간</span></div>
        <div class="metric"><strong id="mFrontiers">-</strong><span>frontier 후보 수</span></div>
        <div class="metric"><strong id="mEvents">-</strong><span>목표 발견 로그</span></div>
      </div>
      <div class="grid">
        <div class="card">
          <h2>탐사 지도</h2>
          <div class="map-wrap">
            <canvas id="explorationMap" width="880" height="720"></canvas>
          </div>
          <div class="legend">
            <span><i class="swatch" style="background:#2f3439"></i>미탐사</span>
            <span><i class="swatch" style="background:#f4efe5"></i>탐사한 이동 가능 영역</span>
            <span><i class="swatch" style="background:#141719"></i>관측된 장애물</span>
            <span><i class="swatch" style="background:#2c82d9"></i>로봇 이동 경로</span>
            <span><i class="swatch" style="background:#19a76f"></i>현재 로봇</span>
          </div>
        </div>
        <div class="card">
          <h2>로봇 카메라 화면</h2>
          <div class="camera-wrap">
            <img id="cameraImage" alt="아직 카메라 화면이 없습니다">
          </div>
        </div>
        <div class="card">
          <h2>현재 탐색 상태</h2>
          <div class="card-body">
            <div id="simpleStatus" class="simple-log">
              <div>대시보드 초기화 중...</div>
            </div>
          </div>
        </div>
        <div class="card">
          <h2>실행 로그</h2>
          <div class="card-body">
            <div id="runLog" class="path"></div>
          </div>
        </div>
        <div class="card">
          <h2>목표 발견 로그</h2>
          <div class="card-body">
            <div id="eventsPath" class="path"></div>
            <div style="overflow:auto; margin-top:10px;">
              <table id="eventsTable"></table>
            </div>
          </div>
        </div>
        <div class="card hidden-debug">
          <h2>관측 voxel CSV</h2>
          <div class="card-body">
            <div id="voxelPath" class="path"></div>
            <div style="overflow:auto; margin-top:10px;">
              <table id="voxelTable"></table>
            </div>
          </div>
        </div>
        <div class="card hidden-debug">
          <h2>Gazebo 탐색 지표</h2>
          <div class="card-body">
            <div id="metricsPath" class="path"></div>
            <div style="overflow:auto; margin-top:10px;">
              <table id="metricsTable"></table>
            </div>
            <div id="trajPath" class="path"></div>
            <div style="overflow:auto; margin-top:10px;">
              <table id="trajTable"></table>
            </div>
          </div>
        </div>
        <div class="card hidden-debug">
          <h2>기존 실험 요약 결과</h2>
          <div class="card-body" id="aggregateBox"></div>
        </div>
      </div>
    </main>
  </div>
  <script>
    const requiredTopics = ['/camera/image_raw', '/camera/depth/image_raw', '/camera/depth/camera_info', '/odom'];
    const cmdMap = {
      forward: [0.12, 0.0],
      back: [-0.08, 0.0],
      left: [0.04, 0.55],
      right: [0.04, -0.55],
      spin_left: [0.0, 0.65],
      spin_right: [0.0, -0.65],
      stop: [0.0, 0.0],
    };

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      return await res.json();
    }

    function fmt(n) {
      if (n === null || n === undefined) return '-';
      return Number(n).toLocaleString();
    }

    function seconds(s) {
      if (!s) return '0s';
      const m = Math.floor(s / 60);
      const r = Math.floor(s % 60);
      return m ? `${m}m ${r}s` : `${r}s`;
    }

    function renderTable(el, data, maxRows = 12) {
      if (!data.exists || !data.rows.length) {
        el.innerHTML = '<tbody><tr><td>No rows yet</td></tr></tbody>';
        return;
      }
      const fields = data.fields.slice(0, 8);
      const head = '<thead><tr>' + fields.map(f => `<th>${f}</th>`).join('') + '</tr></thead>';
      const body = '<tbody>' + data.rows.slice(0, maxRows).map(row =>
        '<tr>' + fields.map(f => `<td>${row[f] ?? ''}</td>`).join('') + '</tr>'
      ).join('') + '</tbody>';
      el.innerHTML = head + body;
    }

    function renderTopics(topics) {
      const html = requiredTopics.map(t => {
        const ok = topics.includes(t);
        return `<div class="topic">${ok ? '정상' : '대기'} ${t}</div>`;
      }).join('') + topics.filter(t => !requiredTopics.includes(t)).slice(0, 30)
        .map(t => `<div class="topic">${t}</div>`).join('');
      document.getElementById('topicList').innerHTML = html || '<div class="topic">No ROS topics visible</div>';
    }

    function renderViewers(viewers) {
      document.getElementById('viewerList').innerHTML = viewers.slice(0, 40).map(v =>
        `<div class="viewer">${v.name}</div>`
      ).join('') || '<div class="viewer">No viewer files yet</div>';
    }

    function renderAggregates(files) {
      const box = document.getElementById('aggregateBox');
      if (!files.length) {
        box.textContent = 'No aggregate.csv files found';
        return;
      }
      box.innerHTML = files.map(file => {
        const tableId = 'agg_' + Math.random().toString(36).slice(2);
        setTimeout(() => renderTable(document.getElementById(tableId), file.data, 8), 0);
        return `<div class="path">${file.name}</div><div style="overflow:auto; margin:8px 0 14px;"><table id="${tableId}"></table></div>`;
      }).join('');
    }

    function worldToCanvas(map, x, y, w, h) {
      const nx = (x - map.min_x) / (map.max_x - map.min_x);
      const ny = (y - map.min_y) / (map.max_y - map.min_y);
      return [nx * w, h - ny * h];
    }

    function drawExplorationMap(map) {
      const canvas = document.getElementById('explorationMap');
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#2f3439';
      ctx.fillRect(0, 0, w, h);

      const cellW = w / map.cols;
      const cellH = h / map.rows;

      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 1;
      for (let ix = 0; ix <= map.cols; ix += 5) {
        const x = ix * cellW;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let iy = 0; iy <= map.rows; iy += 5) {
        const y = h - iy * cellH;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      ctx.fillStyle = '#f4efe5';
      for (const [ix, iy] of map.free || []) {
        ctx.fillRect(ix * cellW, h - (iy + 1) * cellH, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
      }

      ctx.fillStyle = '#141719';
      for (const [ix, iy] of map.occupied || []) {
        ctx.fillRect(ix * cellW, h - (iy + 1) * cellH, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
      }

      if ((map.path || []).length > 1) {
        ctx.strokeStyle = '#2c82d9';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        map.path.forEach((p, i) => {
          const [x, y] = worldToCanvas(map, p[0], p[1], w, h);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      if (map.robot) {
        const [rx, ry] = worldToCanvas(map, map.robot[0], map.robot[1], w, h);
        const yaw = map.robot[2] || 0;
        ctx.save();
        ctx.translate(rx, ry);
        ctx.rotate(-yaw);
        ctx.fillStyle = '#19a76f';
        ctx.beginPath();
        ctx.moveTo(12, 0);
        ctx.lineTo(-9, -7);
        ctx.lineTo(-9, 7);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(rx, ry, 12, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (!(map.free || []).length && !(map.occupied || []).length) {
        ctx.fillStyle = '#bfc6cc';
        ctx.font = '22px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('아직 탐사된 영역이 없습니다', w / 2, h / 2);
      }
    }

    async function refresh() {
      const data = await api('/api/status');
      document.getElementById('runDot').className = 'dot' + (data.running ? ' on' : '');
      document.getElementById('runText').textContent = data.running ? '실행 중' : '정지됨';
      document.getElementById('startNavBtn').disabled = data.running;
      document.getElementById('llmStartBtn').disabled = data.running;
      document.getElementById('stopBtn').disabled = !data.running;
      document.getElementById('mObserved').textContent = fmt(data.voxels.total);
      document.getElementById('mOccupied').textContent = fmt(data.voxels.occupied);
      document.getElementById('mUptime').textContent = seconds(data.uptime);
      const latestMetric = data.metrics_rows.rows[0] || {};
      document.getElementById('mFrontiers').textContent = latestMetric.frontier_count ?? '-';
      document.getElementById('mEvents').textContent = latestMetric.target_event_count ?? '-';
      document.getElementById('voxelPath').textContent = data.voxels.path;
      document.getElementById('metricsPath').textContent = data.metrics_rows.path;
      document.getElementById('trajPath').textContent = data.trajectory_rows.path;
      document.getElementById('eventsPath').textContent = data.target_events_rows.path;
      renderTopics(data.topics);
      renderViewers(data.viewers);
      renderTable(document.getElementById('voxelTable'), data.voxel_rows, 20);
      renderTable(document.getElementById('metricsTable'), data.metrics_rows, 4);
      renderTable(document.getElementById('trajTable'), data.trajectory_rows, 12);
      renderTable(document.getElementById('eventsTable'), data.target_events_rows, 12);
      renderAggregates(data.aggregates);
      drawExplorationMap(data.exploration_map);
      renderSimpleStatus(data);
      renderRunLog(data.log || []);
      const cam = document.getElementById('cameraImage');
      cam.src = '/api/camera.jpg?t=' + Date.now();
    }

    function renderRunLog(lines) {
      const tail = lines.slice(-12);
      document.getElementById('runLog').innerHTML = tail.length
        ? tail.map(line => `<div>${line}</div>`).join('')
        : '<div>아직 실행 로그가 없습니다</div>';
    }

    function renderSimpleStatus(data) {
      const latestMetric = data.metrics_rows.rows[0] || {};
      const runningText = data.running ? '실험 실행 중' : '실험 정지됨';
      const goal = latestMetric.target_goal || document.getElementById('goalSelect').value;
      const eventCount = latestMetric.target_event_count || '0';
      const coverage = latestMetric.coverage_ratio ? `${(Number(latestMetric.coverage_ratio) * 100).toFixed(1)}%` : '-';
      const targetVisible = Number(eventCount) > 0 ? `목표 로그 ${eventCount}건` : '순찰하며 목표 탐색 중';
      const camera = data.camera_exists ? '카메라 화면 수신 중' : '카메라 화면 대기 중';
      const mapping = Number(data.voxels.total || 0) > 0 ? 'voxel map 생성 중' : 'voxel map 대기 중';
      const mappingHint = data.running && data.uptime > 15 && Number(data.voxels.total || 0) === 0
        ? 'voxel map 대기 중: 실행 로그에서 camera_info/odom/depth 수신 상태를 확인하세요'
        : mapping;
      const lines = [
        ['상태', runningText],
        ['목표', goal],
        ['카메라', camera],
        ['맵', mappingHint],
        ['순찰', `coverage ${coverage}, ${targetVisible}`],
        ['최근 로그', data.clean_log || '대기 중'],
      ];
      document.getElementById('simpleStatus').innerHTML = lines.map(([k, v]) => `<div><b>${k}</b><br>${v}</div>`).join('');
    }

    async function post(path, body = {}) {
      return await api(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
    }

    document.getElementById('startNavBtn').addEventListener('click', async () => {
      const goal = document.getElementById('goalSelect').value;
      const res = await post('/api/start', {launch_file: 'rgbd_frontier_navigation.launch.py', launch_args: {target_goal: goal, mission_mode: 'coverage_patrol', stop_on_target: 'false'}});
      document.getElementById('llmHint').textContent = res.ok ? 'Gazebo를 시작했습니다. mapper가 먼저 뜨고 몇 초 뒤 navigator가 움직입니다.' : res.message;
      setTimeout(refresh, 700);
    });
    document.getElementById('llmStartBtn').addEventListener('click', async () => {
      const command = document.getElementById('commandInput').value;
      const interpreted = await post('/api/llm_command', {command});
      const hint = document.getElementById('llmHint');
      if (!interpreted.ok) {
        hint.textContent = interpreted.reason || '명령을 해석하지 못했습니다.';
        return;
      }
      document.getElementById('goalSelect').value = interpreted.goal;
      hint.textContent = `${interpreted.label} 발견 로그를 남기는 순찰 임무로 해석했습니다. Gazebo를 시작합니다.`;
      const res = await post('/api/start', {launch_file: 'rgbd_frontier_navigation.launch.py', launch_args: {target_goal: interpreted.goal, mission_mode: interpreted.mission_mode || 'coverage_patrol', stop_on_target: 'false'}});
      hint.textContent = res.ok
        ? `${interpreted.label} 순찰을 시작했습니다. mapper가 먼저 뜨고 몇 초 뒤 navigator가 움직입니다.`
        : res.message;
      setTimeout(refresh, 700);
    });
    document.getElementById('stopBtn').addEventListener('click', async () => {
      await post('/api/stop');
      setTimeout(refresh, 700);
    });
    for (const btn of document.querySelectorAll('[data-cmd]')) {
      btn.addEventListener('click', async () => {
        const [linear, angular] = cmdMap[btn.dataset.cmd];
        await post('/api/cmd_vel', {linear, angular});
        refresh();
      });
    }
    refresh();
    setInterval(refresh, 2500);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            voxels = observed_voxel_summary()
            self.send_json({
                "running": STATE.is_running(),
                "uptime": STATE.uptime(),
                "log": STATE.snapshot_log(),
                "topics": topic_list(),
                "voxels": voxels,
                "voxel_rows": read_csv_rows(DEFAULT_OUTPUT, max_rows=40),
                "trajectory_rows": read_csv_rows(DEFAULT_TRAJECTORY, max_rows=40),
                "metrics_rows": read_csv_rows(DEFAULT_METRICS, max_rows=10),
                "target_events_rows": read_csv_rows(DEFAULT_TARGET_EVENTS, max_rows=30),
                "graph_nodes_rows": read_csv_rows(DEFAULT_GRAPH_NODES, max_rows=20),
                "graph_edges_rows": read_csv_rows(DEFAULT_GRAPH_EDGES, max_rows=20),
                "frontier_features_rows": read_csv_rows(DEFAULT_FRONTIER_FEATURES, max_rows=20),
                "exploration_map": exploration_map(),
                "camera_exists": DEFAULT_CAMERA.exists(),
                "clean_log": clean_log_line(STATE.snapshot_log()),
                "viewers": list_viewers(),
                "aggregates": aggregate_files(),
            })
            return
        if parsed.path == "/api/read_csv":
            params = parse_qs(parsed.query)
            path = params.get("path", [""])[0]
            self.send_json(read_csv_rows(path, max_rows=80))
            return
        if parsed.path == "/api/camera.jpg":
            if not DEFAULT_CAMERA.exists():
                self.send_response(404)
                self.end_headers()
                return
            body = DEFAULT_CAMERA.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self.read_json()
        if parsed.path == "/api/start":
            launch_args = payload.get("launch_args", {})
            if "target_goal" in launch_args:
                interpreted = infer_goal_from_command(str(launch_args["target_goal"]))
                if interpreted["ok"]:
                    launch_args["target_goal"] = interpreted["goal"]
            ok, message = STATE.start(
                payload.get("launch_file", "rgbd_voxel_mapping.launch.py"),
                launch_args,
            )
            self.send_json({"ok": ok, "message": message})
            return
        if parsed.path == "/api/llm_command":
            self.send_json(infer_goal_from_command(payload.get("command", "")))
            return
        if parsed.path == "/api/stop":
            ok, message = STATE.stop()
            self.send_json({"ok": ok, "message": message})
            return
        if parsed.path == "/api/cmd_vel":
            linear = float(payload.get("linear", 0.0))
            angular = float(payload.get("angular", 0.0))
            code, output = publish_cmd_vel(linear, angular)
            self.send_json({"ok": code == 0, "code": code, "output": output})
            return
        self.send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Gazebo experiment dashboard: http://{args.host}:{args.port}")
    print(f"Workspace: {WORKSPACE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
