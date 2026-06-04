#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
MODE="${1:-dashboard}"
GOAL="${GOAL:-chair}"

cd "${ROOT_DIR}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONHOME PYTHONPATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:${PATH}

set +u
source /opt/ros/humble/setup.bash
set -u

if [[ ! -f "install/setup.bash" ]]; then
  echo "[setup] install/setup.bash not found. Building s_nav_core first."
  colcon build --packages-select s_nav_core --cmake-args -DCMAKE_BUILD_TYPE=Release
fi

set +u
source install/setup.bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:${PATH}

if ! ros2 pkg executables s_nav_core | grep -q "rgbd_frontier_navigator"; then
  echo "[setup] rgbd_frontier_navigator executable not found. Rebuilding s_nav_core."
  colcon build --packages-select s_nav_core --cmake-args -DCMAKE_BUILD_TYPE=Release
  set +u
  source install/setup.bash
  set -u
fi

mkdir -p results/gazebo_rgbd

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[warning] DISPLAY is not set."
  echo "[warning] Gazebo RGB-D camera/depth sensors require rendering."
  echo "[warning] Run this script from a MobaXterm/X11 terminal where the Gazebo GUI can open."
fi

echo "[cleanup] stopping stale Gazebo/ROS experiment processes"
pkill -9 -f 'ros2 launch s_nav_core' >/dev/null 2>&1 || true
pkill -9 -f 'rgbd_frontier_navigator' >/dev/null 2>&1 || true
pkill -9 -f 'depth_voxel_mapper' >/dev/null 2>&1 || true
pkill -9 -f 'robot_state_publisher' >/dev/null 2>&1 || true
pkill -9 -f '/spawn_entity.py' >/dev/null 2>&1 || true
pkill -9 -f 'gzserver' >/dev/null 2>&1 || true
pkill -9 -f 'gzclient' >/dev/null 2>&1 || true
pkill -9 -f 'gazebo --verbose' >/dev/null 2>&1 || true
if command -v fuser >/dev/null 2>&1; then
  fuser -k 11345/tcp 11346/tcp >/dev/null 2>&1 || true
fi

export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11346}"

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi

echo "[dashboard] starting http://${HOST}:${PORT}"
python3 -u scripts/gazebo_experiment_dashboard.py --host "${HOST}" --port "${PORT}" &
DASH_PID=$!

cleanup() {
  echo
  echo "[shutdown] stopping dashboard and Gazebo launch"
  curl -s -X POST "http://${HOST}:${PORT}/api/stop" \
    -H "Content-Type: application/json" \
    -d '{}' >/dev/null 2>&1 || true
  kill "${DASH_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 40); do
  if curl -s "http://${HOST}:${PORT}/api/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

case "${MODE}" in
  nav|navigation|dashboard|ui)
    echo "[experiment] dashboard ready. Enter a natural-language command, then click '자연어 명령 해석 후 시작'."
    ;;
  auto-nav|autonav|auto)
    echo "[experiment] starting Gazebo RGB-D coverage patrol immediately"
    curl -s -X POST "http://127.0.0.1:${PORT}/api/start" \
      -H "Content-Type: application/json" \
      -d "{\"launch_file\":\"rgbd_frontier_navigation.launch.py\",\"launch_args\":{\"target_goal\":\"${GOAL}\",\"mission_mode\":\"coverage_patrol\",\"stop_on_target\":\"false\"}}" >/dev/null
    ;;
  mapping|map)
    echo "[experiment] starting Gazebo RGB-D mapping only"
    curl -s -X POST "http://127.0.0.1:${PORT}/api/start" \
      -H "Content-Type: application/json" \
      -d '{"launch_file":"rgbd_voxel_mapping.launch.py"}' >/dev/null
    ;;
  *)
    echo "Usage: $0 [dashboard|nav|auto-nav|mapping]"
    exit 2
    ;;
esac

if command -v xdg-open >/dev/null 2>&1; then
  if [[ "${HOST}" == "127.0.0.1" || "${HOST}" == "localhost" ]]; then
    xdg-open "http://${HOST}:${PORT}" >/dev/null 2>&1 || true
  fi
fi

echo
echo "Dashboard: http://${HOST}:${PORT}"
if [[ "${HOST}" == "0.0.0.0" ]]; then
  echo "Remote URLs:"
  for ip in $(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.'); do
    echo "  http://${ip}:${PORT}"
  done
  echo "Use the server IP reachable from your laptop."
fi
echo
echo "Modes:"
echo "  ./run_gazebo_dashboard.sh            # dashboard only; start from LLM command button"
echo "  ./run_gazebo_dashboard.sh nav        # same as dashboard only"
echo "  GOAL=bed ./run_gazebo_dashboard.sh auto-nav"
echo "  ./run_gazebo_dashboard.sh mapping    # dashboard + Gazebo + mapper only"
echo "  ./run_gazebo_dashboard.sh dashboard  # dashboard only"
echo
echo "Press Ctrl+C here to stop the dashboard and running experiment."

wait "${DASH_PID}"
