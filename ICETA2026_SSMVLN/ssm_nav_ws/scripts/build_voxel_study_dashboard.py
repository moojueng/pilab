#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from voxel_nav_common import find_target_value, load_voxel_grid, shape3, target_value_for_name


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Voxel Frontier Study Dashboard</title>
  <style>
    :root {
      --paper: #f4f1ea;
      --surface: #fffdf8;
      --surface-2: #f9f6ef;
      --ink: #20262d;
      --muted: #66717b;
      --line: #d9d1c3;
      --teal: #126d6f;
      --blue: #226fc2;
      --red: #c7362f;
      --amber: #d7832f;
      --green: #16875a;
      --target-blue: #246fd3;
      --dark: #1f2429;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 336px minmax(0, 1fr);
    }
    aside {
      background: var(--surface);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(520px, 1fr) auto;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.14;
      letter-spacing: 0;
    }
    .sub {
      margin: 8px 0 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .section {
      border-top: 1px solid var(--line);
      padding: 15px 0;
    }
    .section:first-of-type { border-top: 0; padding-top: 2px; }
    .section-title {
      margin: 0 0 10px;
      color: #4d5862;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .command-box {
      display: grid;
      gap: 8px;
    }
    .command-box textarea {
      width: 100%;
      min-height: 78px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
      font-size: 13px;
      line-height: 1.38;
    }
    .command-box .run {
      background: var(--teal);
      border-color: var(--teal);
      color: #fff;
    }
    .command-status {
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      padding: 10px;
      min-height: 72px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1;
      margin-bottom: 7px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    select, input[type="range"] {
      width: 100%;
    }
    select {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
      font-size: 13px;
    }
    .segmented {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 36px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      font-weight: 750;
      cursor: pointer;
      white-space: nowrap;
    }
    button.active {
      background: var(--teal);
      border-color: var(--teal);
      color: #fff;
    }
    .checks {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 10px;
    }
    label {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 26px;
      font-size: 13px;
      color: #303a43;
      user-select: none;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--teal);
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(260px, 1.2fr) repeat(4, minmax(120px, .45fr));
      gap: 1px;
      border-bottom: 1px solid var(--line);
      background: var(--line);
    }
    .topcell {
      background: rgba(255,253,248,.92);
      padding: 13px 15px;
      min-width: 0;
    }
    .topcell b {
      display: block;
      font-size: 18px;
      line-height: 1;
      margin-bottom: 5px;
    }
    .topcell span {
      color: var(--muted);
      font-size: 12px;
    }
    .scene-wrap {
      position: relative;
      min-width: 0;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(31,36,41,.06) 1px, transparent 1px),
        linear-gradient(rgba(31,36,41,.055) 1px, transparent 1px),
        #ece7dc;
      background-size: 34px 34px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 520px;
      cursor: grab;
    }
    canvas:active { cursor: grabbing; }
    .bottom {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 1px;
      border-top: 1px solid var(--line);
      background: var(--line);
    }
    .panel {
      min-width: 0;
      background: var(--surface);
      padding: 13px 15px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }
    .bars {
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr) 62px;
      gap: 9px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }
    .track {
      height: 8px;
      border-radius: 999px;
      background: #e5ded2;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      width: 0%;
      background: var(--teal);
    }
    .mini-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .mini-table th, .mini-table td {
      padding: 7px 6px;
      border-bottom: 1px solid #e5ded4;
      text-align: left;
      white-space: nowrap;
    }
    .mini-table th {
      color: #59636c;
      font-weight: 800;
    }
    .status-line {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: #323b44;
    }
    .status-line div {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid #ebe4d9;
      padding-bottom: 7px;
    }
    .status-line b { font-size: 12px; color: var(--muted); }
    @media (max-width: 1080px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      main { grid-template-rows: auto 520px auto; }
      .topbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .topcell:first-child { grid-column: span 2; }
      .bottom { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      .metric-grid, .checks, .topbar { grid-template-columns: 1fr; }
      .topcell:first-child { grid-column: auto; }
      .bar-row { grid-template-columns: 82px minmax(0, 1fr) 54px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Voxel Frontier Study</h1>
      <p class="sub" id="missionLine"></p>

      <div class="section">
        <p class="section-title">Natural Language Command</p>
        <div class="command-box">
          <textarea id="commandInput"></textarea>
          <button id="runCommand" class="run">Run Command</button>
          <div id="commandStatus" class="command-status"></div>
        </div>
      </div>

      <div class="section">
        <p class="section-title">Study</p>
        <div class="metric-grid">
          <div class="metric"><strong id="mSuccess">-</strong><span>4-map SSM success</span></div>
          <div class="metric"><strong id="mSteps">-</strong><span>4-map SSM avg target step</span></div>
          <div class="metric"><strong id="mTop1">-</strong><span>frontier top-1</span></div>
          <div class="metric"><strong id="mMaps">-</strong><span>unseen maps</span></div>
        </div>
      </div>

      <div class="section">
        <p class="section-title">Mode</p>
        <div class="segmented" id="modeButtons"></div>
      </div>

      <div class="section">
        <p class="section-title">Map</p>
        <select id="mapSelect"></select>
      </div>

      <div class="section">
        <p class="section-title">Layers</p>
        <div class="checks">
          <label><input id="showObstacles" type="checkbox" checked>Obstacles</label>
          <label><input id="showObserved" type="checkbox" checked>Observed</label>
          <label><input id="showTarget" type="checkbox" checked>Command target</label>
          <label><input id="showPath" type="checkbox" checked>Path</label>
          <label><input id="showSight" type="checkbox" checked>Sight line</label>
          <label><input id="showUnknown" type="checkbox">Unknown</label>
        </div>
      </div>

      <div class="section">
        <p class="section-title">Timeline</p>
        <input id="stepSlider" type="range" min="0" max="0" value="0">
        <div class="segmented" style="margin-top:10px;">
          <button id="prevStep">&lt;</button>
          <button id="playStep" class="active">Play</button>
        </div>
      </div>

      <div class="section">
        <p class="section-title">Current Case</p>
        <div class="status-line" id="caseStatus"></div>
      </div>
    </aside>

    <main>
      <div class="topbar">
        <div class="topcell"><b id="caseTitle">-</b><span id="caseSubtitle">-</span></div>
        <div class="topcell"><b id="kSuccess">-</b><span>success</span></div>
        <div class="topcell"><b id="kStep">-</b><span>target seen step</span></div>
        <div class="topcell"><b id="kRevisit">-</b><span>revisit ratio</span></div>
        <div class="topcell"><b id="kNodes">-</b><span>observed nodes</span></div>
      </div>
      <div class="scene-wrap">
        <canvas id="scene"></canvas>
      </div>
      <div class="bottom">
        <div class="panel">
          <h2>Mode Aggregate</h2>
          <div class="bars" id="bars"></div>
        </div>
        <div class="panel">
          <h2>Map Runs</h2>
          <table class="mini-table" id="runTable"></table>
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
      sight: document.getElementById('showSight'),
      slider: document.getElementById('stepSlider'),
      play: document.getElementById('playStep')
    };

    let mode = DATA.modes.includes('ssm_utility') ? 'ssm_utility' : DATA.modes[0];
    let mapName = DATA.maps[0];
    let step = 0;
    let yaw = -0.72;
    let pitch = 0.66;
    let dragging = false;
    let last = {x: 0, y: 0};
    let timer = null;

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || value === '') return '-';
      const n = Number(value);
      if (!Number.isFinite(n)) return String(value);
      return n.toFixed(digits).replace(/\.00$/, '');
    }

    function pct(value) {
      return `${fmt(Number(value) * 100, 1)}%`;
    }

    function caseData() {
      return DATA.cases[mode][mapName];
    }

    function setupControls() {
      document.getElementById('missionLine').textContent = DATA.mission.raw_command || DATA.mission.target_name || '';
      document.getElementById('commandInput').value = DATA.mission.raw_command || '';
      document.getElementById('runCommand').addEventListener('click', runNaturalLanguageCommand);
      const modeBox = document.getElementById('modeButtons');
      modeBox.innerHTML = DATA.modes.map(m => `<button data-mode="${m}">${m}</button>`).join('');
      for (const btn of modeBox.querySelectorAll('button')) {
        btn.addEventListener('click', () => {
          mode = btn.dataset.mode;
          refreshCase(true);
        });
      }
      const mapSelect = document.getElementById('mapSelect');
      mapSelect.innerHTML = DATA.maps.map(m => `<option value="${m}">${m}</option>`).join('');
      mapSelect.value = mapName;
      mapSelect.addEventListener('change', () => {
        mapName = mapSelect.value;
        refreshCase(true);
      });
      for (const item of [controls.obstacles, controls.observed, controls.unknown, controls.target, controls.path, controls.sight]) {
        item.addEventListener('change', draw);
      }
      controls.slider.addEventListener('input', e => {
        step = Number(e.target.value);
        draw();
      });
      document.getElementById('prevStep').addEventListener('click', () => {
        step = Math.max(0, step - 1);
        draw();
      });
      controls.play.addEventListener('click', togglePlay);
      canvas.addEventListener('mousedown', e => {
        dragging = true;
        last = {x: e.clientX, y: e.clientY};
      });
      window.addEventListener('mouseup', () => dragging = false);
      window.addEventListener('mousemove', e => {
        if (!dragging) return;
        yaw += (e.clientX - last.x) * 0.008;
        pitch += (e.clientY - last.y) * 0.008;
        pitch = Math.max(-1.22, Math.min(1.22, pitch));
        last = {x: e.clientX, y: e.clientY};
        draw();
      });
      window.addEventListener('resize', resize);
    }

    async function runNaturalLanguageCommand() {
      const status = document.getElementById('commandStatus');
      const button = document.getElementById('runCommand');
      const command = document.getElementById('commandInput').value.trim();
      if (!command) {
        status.textContent = '명령문을 입력하세요.';
        return;
      }
      if (window.location.protocol === 'file:') {
        status.textContent = '실행 버튼은 로컬 서버에서 열었을 때 동작합니다. 터미널에서 python3 scripts/voxel_study_dashboard_server.py 실행 후 http://127.0.0.1:8787 로 접속하세요.';
        return;
      }
      button.disabled = true;
      button.textContent = 'Running...';
      status.textContent = '명령을 grounding하고 4개 unseen voxel map 평가를 다시 실행 중입니다.';
      try {
        const res = await fetch('/api/run_command', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({command})
        });
        const data = await res.json();
        if (!data.ok) {
          status.textContent = data.message || '실행 실패';
          button.disabled = false;
          button.textContent = 'Run Command';
          return;
        }
        status.textContent = '완료. 새 결과를 불러옵니다.';
        window.location.reload();
      } catch (err) {
        status.textContent = `서버 연결 실패: ${err}`;
        button.disabled = false;
        button.textContent = 'Run Command';
      }
    }

    function updateStudyMetrics() {
      const ssm = DATA.aggregate.find(r => r.mode === 'ssm_utility') || DATA.aggregate[0] || {};
      document.getElementById('mSuccess').textContent = pct(ssm.success_rate || 0);
      document.getElementById('mSteps').textContent = fmt(ssm.avg_steps_success || 0, 2);
      document.getElementById('mTop1').textContent = DATA.training.best_top1 ? pct(DATA.training.best_top1) : '-';
      document.getElementById('mMaps').textContent = `${DATA.maps.length}`;
    }

    function updateBars() {
      const maxSteps = Math.max(...DATA.aggregate.map(r => Number(r.avg_steps_success || 0)), 1);
      document.getElementById('bars').innerHTML = DATA.aggregate.map(row => {
        const steps = Number(row.avg_steps_success || 0);
        const width = Math.max(4, Math.min(100, (steps / maxSteps) * 100));
        const color = row.mode === 'ssm_utility' ? 'var(--teal)' : 'var(--amber)';
        return `
          <div class="bar-row"><span>${row.mode}</span><div class="track"><div class="fill" style="width:${width}%;background:${color}"></div></div><span>${fmt(steps, 2)}</span></div>
          <div class="bar-row"><span>revisit</span><div class="track"><div class="fill" style="width:${Math.min(100, Number(row.avg_revisit_ratio || 0) * 100)}%;background:var(--red)"></div></div><span>${pct(row.avg_revisit_ratio || 0)}</span></div>
        `;
      }).join('');
    }

    function updateRunTable() {
      const rows = DATA.summary.filter(r => r.mode === mode);
      const head = '<thead><tr><th>map</th><th>acc</th><th>target step</th><th>revisit</th><th>nodes</th></tr></thead>';
      const body = '<tbody>' + rows.map(r => `
        <tr>
          <td>${r.map}</td>
          <td>${Number(r.success || 0) ? '100%' : '0%'}</td>
          <td>${r.target_seen_step ?? r.steps}</td>
          <td>${fmt(Number(r.revisit_ratio) * 100, 1)}%</td>
          <td>${r.observed_nodes}</td>
        </tr>
      `).join('') + '</tbody>';
      document.getElementById('runTable').innerHTML = head + body;
    }

    function refreshCase(resetStep) {
      document.querySelectorAll('[data-mode]').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
      document.getElementById('mapSelect').value = mapName;
      const cd = caseData();
      const maxStep = Math.max(0, cd.trajectory.length - 1);
      controls.slider.max = maxStep;
      if (resetStep) step = maxStep;
      step = Math.max(0, Math.min(maxStep, step));
      document.getElementById('caseTitle').textContent = `${mapName} / ${mode}`;
      document.getElementById('caseSubtitle').textContent = `target=${DATA.mission.target_name || 'target'} action=${DATA.mission.on_detection_action || 'log'}`;
      document.getElementById('kSuccess').textContent = Number(cd.metrics.success || 0) ? 'yes' : 'no';
      document.getElementById('kStep').textContent = cd.metrics.target_seen_step ?? '-';
      document.getElementById('kRevisit').textContent = pct(cd.metrics.revisit_ratio || 0);
      document.getElementById('kNodes').textContent = cd.metrics.observed_nodes || '-';
      document.getElementById('caseStatus').innerHTML = [
        ['grounding', DATA.mission.grounding_type || '-'],
        ['llm_used', String(Boolean(DATA.mission.llm_used))],
        ['fallbacks', cd.metrics.fallback_count || '0'],
        ['frontier switches', cd.metrics.frontier_switches || '0'],
        ['depth rays', cd.metrics.depth_rays || '0']
      ].map(([k, v]) => `<div><b>${k}</b><span>${v}</span></div>`).join('');
      updateRunTable();
      updateBars();
      resize();
    }

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

    function centered(v, cd) {
      return {
        x: v.c - (cd.cols - 1) / 2,
        y: -(v.z - (cd.depth - 1) / 2),
        z: v.r - (cd.rows - 1) / 2
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

    function project(v, cd) {
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width / Math.max(cd.cols, 1), rect.height / Math.max(cd.rows + cd.depth, 1)) * 0.84;
      const rp = rotate(centered(v, cd));
      return {
        x: rect.width / 2 + rp.x * scale,
        y: rect.height / 2 + rp.y * scale,
        depth: rp.z,
        scale
      };
    }

    function color(hex, shade, alpha) {
      const n = parseInt(hex.slice(1), 16);
      const r = Math.round(((n >> 16) & 255) * shade);
      const g = Math.round(((n >> 8) & 255) * shade);
      const b = Math.round((n & 255) * shade);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    function cubeFaces(v, cd, hex, alpha, size) {
      const h = size / 2;
      const corners = [
        {c:v.c-h,r:v.r-h,z:v.z-h},{c:v.c+h,r:v.r-h,z:v.z-h},{c:v.c+h,r:v.r+h,z:v.z-h},{c:v.c-h,r:v.r+h,z:v.z-h},
        {c:v.c-h,r:v.r-h,z:v.z+h},{c:v.c+h,r:v.r-h,z:v.z+h},{c:v.c+h,r:v.r+h,z:v.z+h},{c:v.c-h,r:v.r+h,z:v.z+h}
      ];
      const faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]];
      return faces.map((idx, i) => {
        const pts = idx.map(j => project(corners[j], cd));
        const depth = pts.reduce((s, p) => s + p.depth, 0) / pts.length;
        const shade = [0.9, 1, 0.94, 0.78, 0.84, 0.72][i];
        return {pts, depth, hex, alpha, shade};
      });
    }

    function drawFace(face) {
      ctx.beginPath();
      ctx.moveTo(face.pts[0].x, face.pts[0].y);
      for (const p of face.pts.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.closePath();
      ctx.fillStyle = color(face.hex, face.shade, face.alpha);
      ctx.fill();
      ctx.strokeStyle = 'rgba(31,36,41,.16)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    function drawLine(points, cd, style, width, dashed = false) {
      if (points.length < 2) return;
      const first = project(points[0], cd);
      ctx.beginPath();
      ctx.moveTo(first.x, first.y);
      for (const p of points.slice(1)) {
        const pp = project(p, cd);
        ctx.lineTo(pp.x, pp.y);
      }
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.setLineDash(dashed ? [8, 7] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawAxes(cd) {
      const origin = {z: cd.depth - 1, r: cd.rows - 1, c: 0};
      drawLine([origin, {z: origin.z, r: origin.r, c: cd.cols - 1}], cd, 'rgba(18,109,111,.8)', 2);
      drawLine([origin, {z: origin.z, r: 0, c: origin.c}], cd, 'rgba(34,111,194,.75)', 2);
      drawLine([origin, {z: 0, r: origin.r, c: origin.c}], cd, 'rgba(215,131,47,.8)', 2);
    }

    function drawSight(cd, pathNow) {
      const targetSeenStep = Number(cd.metrics.target_seen_step ?? -1);
      if (!controls.sight.checked || targetSeenStep < 0 || step < targetSeenStep || !cd.target) return;
      const seen = cd.trajectory[Math.min(targetSeenStep, cd.trajectory.length - 1)];
      if (!seen) return;
      drawLine([seen, cd.target], cd, 'rgba(215,131,47,.95)', 3.4, true);
    }

    function drawCommandTargetMarker(cd) {
      if (!controls.target.checked || !cd.target) return;
      const p = project(cd.target, cd);
      const color = commandTargetColor();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255,255,255,.95)';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
    }

    function draw() {
      const cd = caseData();
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      drawAxes(cd);
      const faces = [];
      if (controls.unknown.checked) {
        for (const v of cd.observed) if (v.value === -1) faces.push(...cubeFaces(v, cd, '#bbb6ae', .11, .72));
      }
      if (controls.observed.checked) {
        for (const v of cd.observed) if (v.value === 0) faces.push(...cubeFaces(v, cd, '#f7f5ee', .43, .74));
      }
      if (controls.obstacles.checked) {
        for (const v of cd.mapVoxels) if (v.value === 1) faces.push(...cubeFaces(v, cd, '#1f2429', .56, .82));
      }
      if (controls.target.checked) {
        for (const v of cd.mapVoxels) {
          if (v.value === cd.targetValue) {
            faces.push(...cubeFaces(v, cd, commandTargetColor(), .96, 1.06));
          }
        }
      }
      faces.sort((a, b) => a.depth - b.depth).forEach(drawFace);
      const pathNow = cd.trajectory.slice(0, step + 1);
      drawSight(cd, pathNow);
      drawCommandTargetMarker(cd);
      if (controls.path.checked) {
        drawLine(pathNow, cd, 'rgba(34,111,194,.95)', 4);
        for (let i = 0; i < pathNow.length; i++) {
          const p = project(pathNow[i], cd);
          ctx.beginPath();
          ctx.arc(p.x, p.y, i === pathNow.length - 1 ? 7.2 : 4.1, 0, Math.PI * 2);
          ctx.fillStyle = i === pathNow.length - 1 ? '#16875a' : '#226fc2';
          ctx.fill();
          ctx.strokeStyle = 'rgba(255,255,255,.88)';
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }
      }
      controls.slider.value = step;
    }

    function targetColor(value) {
      if (value === 2) return '#c7362f';
      if (value === 3) return '#246fd3';
      if (value === 4) return '#16875a';
      return '#d7832f';
    }

    function commandTargetColor() {
      const color = String(DATA.mission.target_color || '').toLowerCase();
      const table = {
        red: '#c7362f',
        blue: '#246fd3',
        green: '#16875a',
        yellow: '#d39a1e',
        orange: '#d7832f',
        purple: '#7a4bb5',
        pink: '#c45291',
        brown: '#8a5a34',
        black: '#24272c',
        gray: '#69727d',
        grey: '#69727d',
        white: '#f7f5ee',
        any: targetColor(caseData().targetValue),
      };
      return table[color] || targetColor(caseData().targetValue);
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
        const cd = caseData();
        step = step >= cd.trajectory.length - 1 ? 0 : step + 1;
        draw();
      }, 180);
    }

    setupControls();
    updateStudyMetrics();
    refreshCase(true);
  </script>
</body>
</html>
"""


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_trajectory(path):
    return [
        {"step": int(r["step"]), "z": int(r["z"]), "r": int(r["r"]), "c": int(r["c"])}
        for r in read_csv(path)
    ]


def read_observed(path):
    return [
        {
            "z": int(r["z"]),
            "r": int(r["r"]),
            "c": int(r["c"]),
            "value": int(r["value"]),
            "visits": int(r["visits"]),
        }
        for r in read_csv(path)
    ]


def grid_to_voxels(grid):
    voxels = []
    depth, rows, cols = shape3(grid)
    for z in range(depth):
        for r in range(rows):
            for c in range(cols):
                value = grid[z][r][c]
                if value != 0:
                    voxels.append({"z": z, "r": r, "c": c, "value": value})
    return voxels


def find_target_or_none(grid, target_value):
    try:
        return find_target_value(grid, target_value)
    except ValueError:
        return None


def numeric_rows(rows):
    converted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            try:
                if value is not None and value != "" and key not in ("mode", "map"):
                    item[key] = float(value)
                    if item[key].is_integer():
                        item[key] = int(item[key])
                else:
                    item[key] = value
            except ValueError:
                item[key] = value
        converted.append(item)
    return converted


def best_training_top1(path):
    rows = read_csv(path)
    best = 0.0
    for row in rows:
        try:
            best = max(best, float(row.get("best_top1", 0.0)))
        except ValueError:
            pass
    return best


def build_dashboard(args):
    result_dir = Path(args.result_dir)
    map_dir = Path(args.map_dir)
    summary = numeric_rows(read_csv(result_dir / "summary.csv"))
    aggregate = numeric_rows(read_csv(result_dir / "aggregate.csv"))
    mission = read_json(result_dir / "mission.json")
    selected_target_value = target_value_for_name(mission.get("target_name", "red_chair"))
    modes = sorted({row["mode"] for row in summary})
    maps = sorted({row["map"] for row in summary})
    cases = {mode: {} for mode in modes}

    for mode in modes:
        for map_name in maps:
            case_dir = result_dir / mode / map_name
            map_path = map_dir / f"{map_name}.vxl"
            grid = load_voxel_grid(map_path)
            depth, rows, cols = shape3(grid)
            target = find_target_or_none(grid, selected_target_value)
            metrics_rows = numeric_rows(read_csv(case_dir / "metrics.csv"))
            metrics = metrics_rows[0] if metrics_rows else {}
            cases[mode][map_name] = {
                "depth": depth,
                "rows": rows,
                "cols": cols,
                "mapVoxels": grid_to_voxels(grid),
                "observed": read_observed(case_dir / "observed_voxels.csv"),
                "trajectory": read_trajectory(case_dir / "trajectory.csv"),
                "target": {"z": target.z, "r": target.r, "c": target.c} if target else None,
                "targetValue": selected_target_value,
                "metrics": metrics,
            }

    data = {
        "mission": mission,
        "summary": summary,
        "aggregate": aggregate,
        "training": {"best_top1": best_training_top1(result_dir / "train_frontier_ssm_log.csv")},
        "modes": modes,
        "maps": maps,
        "cases": cases,
    }
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Build a single static dashboard for the 3D voxel frontier study.")
    parser.add_argument("--result-dir", default="results/vln_ssm_voxel_study")
    parser.add_argument("--map-dir", default="maps/vln_ssm_voxel_study/voxel_unseen")
    parser.add_argument("--out", default="results/vln_ssm_voxel_study/dashboard.html")
    args = parser.parse_args()
    build_dashboard(args)


if __name__ == "__main__":
    main()
