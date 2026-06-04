#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from voxel_nav_common import find_target, load_voxel_grid, shape3


def read_trajectory(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "step": int(row["step"]),
                "z": int(row["z"]),
                "r": int(row["r"]),
                "c": int(row["c"]),
            })
    return rows


def read_observed(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "z": int(row["z"]),
                "r": int(row["r"]),
                "c": int(row["c"]),
                "value": int(row["value"]),
                "visits": int(row["visits"]),
            })
    return rows


def read_metrics(path):
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def grid_to_voxels(grid):
    voxels = []
    depth, rows, cols = shape3(grid)
    for z in range(depth):
        for r in range(rows):
            for c in range(cols):
                v = grid[z][r][c]
                if v != 0:
                    voxels.append({"z": z, "r": r, "c": c, "value": v})
    return voxels


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3D Voxel Navigation Viewer</title>
  <style>
    :root {
      --bg: #f7f4ee;
      --panel: #ffffff;
      --ink: #1f2428;
      --muted: #6d747c;
      --line: #d8d2c8;
      --accent: #116d6e;
      --orange: #ef6f3e;
      --blue: #1f77d0;
      --green: #18a058;
      --red: #d92d20;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 22px 20px;
    }
    main {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-width: 0;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.18;
      letter-spacing: 0;
    }
    .sub {
      margin: 0 0 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 18px 0;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfaf7;
    }
    .metric b {
      display: block;
      font-size: 18px;
      line-height: 1;
      margin-bottom: 4px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    label {
      display: flex;
      gap: 9px;
      align-items: center;
      margin: 10px 0;
      color: #29323a;
      font-size: 13px;
      user-select: none;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .section {
      padding: 16px 0;
      border-top: 1px solid var(--line);
    }
    .section-title {
      margin: 0 0 10px;
      color: #4b535b;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .legend {
      display: grid;
      gap: 9px;
      font-size: 13px;
      color: #303841;
    }
    .key {
      display: flex;
      align-items: center;
      gap: 9px;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,.18);
    }
    .viewer {
      position: relative;
      min-height: 540px;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(0,0,0,.045) 1px, transparent 1px),
        linear-gradient(rgba(0,0,0,.04) 1px, transparent 1px),
        #f3efe6;
      background-size: 36px 36px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    canvas:active { cursor: grabbing; }
    .hud {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-top: 1px solid var(--line);
      background: rgba(255,255,255,.9);
      backdrop-filter: blur(12px);
    }
    .hud strong {
      font-size: 14px;
    }
    .hud span {
      color: var(--muted);
      font-size: 13px;
    }
    .btn-row {
      display: flex;
      gap: 8px;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--ink);
      min-width: 38px;
      height: 34px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary {
      min-width: 74px;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    @media (max-width: 780px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .viewer { min-height: 440px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>3D Voxel Navigation Viewer</h1>
      <p class="sub">Drag to rotate. Use the step slider to inspect how the agent moved through the voxel map.</p>

      <div class="metric-grid">
        <div class="metric"><b id="metricDims"></b><span>voxel map</span></div>
        <div class="metric"><b id="metricSteps"></b><span>trajectory steps</span></div>
        <div class="metric"><b id="metricCurrent"></b><span>current voxel</span></div>
        <div class="metric"><b id="metricZ"></b><span>z layer</span></div>
      </div>

      <div class="section">
        <p class="section-title">Layers</p>
        <label><input id="showObstacles" type="checkbox" checked>Full-map obstacles</label>
        <label><input id="showObserved" type="checkbox" checked>Observed free voxels</label>
        <label><input id="showUnknown" type="checkbox">Unknown observed-state voxels</label>
        <label><input id="showTarget" type="checkbox" checked>Target voxel</label>
        <label><input id="showPath" type="checkbox" checked>Trajectory path</label>
        <label><input id="showSightLine" type="checkbox" checked>Target sight line</label>
      </div>

      <div class="section">
        <p class="section-title">Step</p>
        <input id="stepSlider" type="range" min="0" max="0" value="0">
      </div>

      <div class="section">
        <p class="section-title">Legend</p>
        <div class="legend">
          <div class="key"><span class="swatch" style="background:#202225"></span>Obstacle</div>
          <div class="key"><span class="swatch" style="background:#f2f2ef"></span>Observed free voxel</div>
          <div class="key"><span class="swatch" style="background:#d92d20"></span>Target</div>
          <div class="key"><span class="swatch" style="background:#1f77d0"></span>Past path</div>
          <div class="key"><span class="swatch" style="background:#18a058"></span>Current robot</div>
          <div class="key"><span class="swatch" style="background:#f59e0b"></span>Target enters view</div>
        </div>
      </div>
    </aside>

    <main>
      <div class="viewer"><canvas id="scene"></canvas></div>
      <div class="hud">
        <div>
          <strong id="caseTitle"></strong><br>
          <span id="stepText"></span>
        </div>
        <div class="btn-row">
          <button id="resetView" title="Reset view">R</button>
          <button id="prevStep" title="Previous step">&lt;</button>
          <button id="play" class="primary">Play</button>
          <button id="nextStep" title="Next step">&gt;</button>
        </div>
      </div>
    </main>
  </div>

  <script>
    const DATA = __DATA__;
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const controls = {
      obstacles: document.getElementById('showObstacles'),
      observed: document.getElementById('showObserved'),
      unknown: document.getElementById('showUnknown'),
      target: document.getElementById('showTarget'),
      path: document.getElementById('showPath'),
      sightLine: document.getElementById('showSightLine'),
      slider: document.getElementById('stepSlider'),
      play: document.getElementById('play')
    };

    let yaw = -0.72;
    let pitch = 0.68;
    let dragging = false;
    let last = {x: 0, y: 0};
    let step = Math.max(0, DATA.trajectory.length - 1);
    let timer = null;

    controls.slider.max = Math.max(0, DATA.trajectory.length - 1);
    controls.slider.value = step;
    document.getElementById('caseTitle').textContent = DATA.title;
    document.getElementById('metricDims').textContent = `${DATA.cols}x${DATA.rows}x${DATA.depth}`;
    document.getElementById('metricSteps').textContent = `${DATA.trajectory.length}`;

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(320, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function centered(v) {
      return {
        x: v.c - (DATA.cols - 1) / 2,
        y: -(v.z - (DATA.depth - 1) / 2),
        z: v.r - (DATA.rows - 1) / 2
      };
    }

    function rotate(p) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = p.x * cy - p.z * sy;
      const z1 = p.x * sy + p.z * cy;
      const y1 = p.y * cp - z1 * sp;
      const z2 = p.y * sp + z1 * cp;
      return {x: x1, y: y1, z: z2};
    }

    function project(p) {
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width / Math.max(DATA.cols, 1), rect.height / Math.max(DATA.rows + DATA.depth, 1)) * 0.82;
      const rp = rotate(centered(p));
      return {
        x: rect.width / 2 + rp.x * scale,
        y: rect.height / 2 + rp.y * scale,
        depth: rp.z,
        scale
      };
    }

    function cubeFaces(v, color, alpha, size) {
      const h = size / 2;
      const corners = [
        {c:v.c-h,r:v.r-h,z:v.z-h},{c:v.c+h,r:v.r-h,z:v.z-h},{c:v.c+h,r:v.r+h,z:v.z-h},{c:v.c-h,r:v.r+h,z:v.z-h},
        {c:v.c-h,r:v.r-h,z:v.z+h},{c:v.c+h,r:v.r-h,z:v.z+h},{c:v.c+h,r:v.r+h,z:v.z+h},{c:v.c-h,r:v.r+h,z:v.z+h}
      ];
      const faceIdx = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]];
      return faceIdx.map((idx, i) => {
        const pts = idx.map(j => project(corners[j]));
        const depth = pts.reduce((s, p) => s + p.depth, 0) / pts.length;
        const shade = [0.88, 1.0, 0.94, 0.78, 0.84, 0.72][i];
        return {pts, depth, color, alpha, shade};
      });
    }

    function colorWithShade(hex, shade, alpha) {
      const n = parseInt(hex.slice(1), 16);
      const r = Math.round(((n >> 16) & 255) * shade);
      const g = Math.round(((n >> 8) & 255) * shade);
      const b = Math.round((n & 255) * shade);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    function drawFace(face) {
      ctx.beginPath();
      ctx.moveTo(face.pts[0].x, face.pts[0].y);
      for (const p of face.pts.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.closePath();
      ctx.fillStyle = colorWithShade(face.color, face.shade, face.alpha);
      ctx.fill();
      ctx.strokeStyle = 'rgba(39, 43, 48, .18)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    function drawLine(points, color, width, dashed = false) {
      if (points.length < 2) return;
      ctx.beginPath();
      const first = project(points[0]);
      ctx.moveTo(first.x, first.y);
      for (const p of points.slice(1)) {
        const pp = project(p);
        ctx.lineTo(pp.x, pp.y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.setLineDash(dashed ? [8, 7] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawSightLine(pathNow) {
      if (!controls.sightLine.checked || DATA.targetSeenStep < 0 || step < DATA.targetSeenStep || !DATA.target) return;
      const seenPose = DATA.trajectory[Math.min(DATA.targetSeenStep, DATA.trajectory.length - 1)];
      if (!seenPose) return;
      drawLine([seenPose, DATA.target], 'rgba(245, 158, 11, .95)', 3.5, true);
      const a = project(seenPose);
      const b = project(DATA.target);
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,.88)';
      ctx.strokeStyle = 'rgba(245, 158, 11, .9)';
      ctx.lineWidth = 1;
      const label = `target seen @ step ${DATA.targetSeenStep}`;
      ctx.font = '700 12px system-ui, sans-serif';
      const width = ctx.measureText(label).width + 16;
      ctx.beginPath();
      ctx.roundRect(mx - width / 2, my - 26, width, 22, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#9a5b00';
      ctx.fillText(label, mx - width / 2 + 8, my - 10);
      ctx.restore();
    }

    function drawAxes() {
      const origin = {z: DATA.depth - 1, r: DATA.rows - 1, c: 0};
      const axes = [
        [{...origin}, {z: origin.z, r: origin.r, c: DATA.cols - 1}, '#116d6e', 'c'],
        [{...origin}, {z: origin.z, r: 0, c: origin.c}, '#8c5fbf', 'r'],
        [{...origin}, {z: 0, r: origin.r, c: origin.c}, '#d46a2c', 'z']
      ];
      for (const [a, b, color, label] of axes) {
        drawLine([a, b], color, 2);
        const end = project(b);
        ctx.fillStyle = color;
        ctx.font = '700 13px system-ui, sans-serif';
        ctx.fillText(label, end.x + 8, end.y + 4);
      }
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      drawAxes();

      const faces = [];
      if (controls.unknown.checked) {
        for (const v of DATA.observed) {
          if (v.value === -1) faces.push(...cubeFaces(v, '#c9c4bc', 0.13, 0.72));
        }
      }
      if (controls.observed.checked) {
        for (const v of DATA.observed) {
          if (v.value === 0) faces.push(...cubeFaces(v, '#f7f6f1', 0.42, 0.74));
        }
      }
      if (controls.obstacles.checked) {
        for (const v of DATA.mapVoxels) {
          if (v.value === 1) faces.push(...cubeFaces(v, '#202225', 0.58, 0.82));
        }
      }
      if (controls.target.checked) {
        for (const v of DATA.mapVoxels) {
          if (v.value === 2) faces.push(...cubeFaces(v, '#d92d20', 0.92, 1.02));
        }
      }

      const pathNow = DATA.trajectory.slice(0, step + 1);
      faces.sort((a, b) => a.depth - b.depth).forEach(drawFace);
      drawSightLine(pathNow);
      if (controls.path.checked) {
        drawLine(pathNow, 'rgba(31,119,208,.95)', 4);
        for (let i = 0; i < pathNow.length; i++) {
          const p = project(pathNow[i]);
          ctx.beginPath();
          ctx.arc(p.x, p.y, i === pathNow.length - 1 ? 7 : 4.2, 0, Math.PI * 2);
          ctx.fillStyle = i === pathNow.length - 1 ? '#18a058' : '#1f77d0';
          ctx.fill();
          ctx.strokeStyle = 'rgba(255,255,255,.9)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      updateText();
    }

    function updateText() {
      const cur = DATA.trajectory[step] || {z: 0, r: 0, c: 0};
      document.getElementById('metricCurrent').textContent = `(${cur.z},${cur.r},${cur.c})`;
      document.getElementById('metricZ').textContent = `${cur.z}`;
      const seen = DATA.targetSeenStep >= 0
        ? ` - target seen at step ${DATA.targetSeenStep}`
        : '';
      document.getElementById('stepText').textContent = `step ${step} / ${Math.max(0, DATA.trajectory.length - 1)} - position z=${cur.z}, r=${cur.r}, c=${cur.c}${seen}`;
      controls.slider.value = step;
    }

    function setStep(next) {
      step = Math.max(0, Math.min(DATA.trajectory.length - 1, next));
      draw();
    }

    function togglePlay() {
      if (timer) {
        clearInterval(timer);
        timer = null;
        controls.play.textContent = 'Play';
        return;
      }
      controls.play.textContent = 'Pause';
      timer = setInterval(() => {
        if (step >= DATA.trajectory.length - 1) setStep(0);
        else setStep(step + 1);
      }, 180);
    }

    canvas.addEventListener('mousedown', e => {
      dragging = true;
      last = {x: e.clientX, y: e.clientY};
    });
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      yaw += (e.clientX - last.x) * 0.008;
      pitch += (e.clientY - last.y) * 0.008;
      pitch = Math.max(-1.25, Math.min(1.25, pitch));
      last = {x: e.clientX, y: e.clientY};
      draw();
    });
    canvas.addEventListener('touchstart', e => {
      if (!e.touches.length) return;
      dragging = true;
      last = {x: e.touches[0].clientX, y: e.touches[0].clientY};
    }, {passive: true});
    canvas.addEventListener('touchmove', e => {
      if (!dragging || !e.touches.length) return;
      yaw += (e.touches[0].clientX - last.x) * 0.008;
      pitch += (e.touches[0].clientY - last.y) * 0.008;
      pitch = Math.max(-1.25, Math.min(1.25, pitch));
      last = {x: e.touches[0].clientX, y: e.touches[0].clientY};
      draw();
    }, {passive: true});
    canvas.addEventListener('touchend', () => dragging = false);

    controls.slider.addEventListener('input', e => setStep(Number(e.target.value)));
    controls.play.addEventListener('click', togglePlay);
    document.getElementById('prevStep').addEventListener('click', () => setStep(step - 1));
    document.getElementById('nextStep').addEventListener('click', () => setStep(step + 1));
    document.getElementById('resetView').addEventListener('click', () => {
      yaw = -0.72;
      pitch = 0.68;
      draw();
    });
    for (const item of [controls.obstacles, controls.observed, controls.unknown, controls.target, controls.path, controls.sightLine]) {
      item.addEventListener('change', draw);
    }
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""


def build_html(args):
    grid = load_voxel_grid(args.map)
    depth, rows, cols = shape3(grid)
    trajectory = read_trajectory(args.trajectory)
    observed = read_observed(args.observed) if args.observed else []
    metrics_path = Path(args.metrics) if args.metrics else Path(args.trajectory).parent / "metrics.csv"
    metrics = read_metrics(metrics_path)
    target = find_target(grid)
    target_seen_step = int(metrics.get("target_seen_step", -1)) if metrics else -1
    title = args.title or f"{Path(args.map).stem} / {Path(args.trajectory).parent.parent.name}"
    data = {
        "title": title,
        "depth": depth,
        "rows": rows,
        "cols": cols,
        "mapVoxels": grid_to_voxels(grid),
        "observed": observed,
        "trajectory": trajectory,
        "target": {"z": target.z, "r": target.r, "c": target.c},
        "targetSeenStep": target_seen_step,
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True, help="Input .vxl map")
    p.add_argument("--trajectory", required=True, help="trajectory.csv from a voxel evaluation case")
    p.add_argument("--observed", help="observed_voxels.csv from the same case")
    p.add_argument("--metrics", help="metrics.csv from the same case. Defaults to trajectory directory / metrics.csv")
    p.add_argument("--out", required=True, help="Output standalone HTML")
    p.add_argument("--title", help="Viewer title")
    args = p.parse_args()
    build_html(args)


if __name__ == "__main__":
    main()
