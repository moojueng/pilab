# VSCode 설정 가이드

이 디렉토리에는 프로젝트를 VSCode에서 효과적으로 사용하기 위한 설정 파일들이 포함되어 있습니다.

## 📁 파일 설명

### settings.json
프로젝트별 VSCode 설정:
- Python 인터프리터 경로 설정 (YOLO venv)
- ROS2 Humble 환경 설정
- C++ IntelliSense 설정
- 파일 연결 및 제외 설정
- 자동 포맷팅 설정

### extensions.json
권장 확장 프로그램 목록:
- Python 개발 도구
- ROS2 개발 도구
- C++ 개발 도구
- YAML, Markdown 편집기
- Git 도구

### launch.json
디버깅 설정:
- YOLO 학습 디버깅
- YOLO 탐지 디버깅 (웹캠/이미지)
- Python 파일 디버깅
- ROS2 Launch 파일 실행

### tasks.json
자동화 작업:
- ROS2 빌드/클린
- TurtleBot3 Gazebo 실행
- Cartographer SLAM 실행
- Navigation2 실행
- RViz2 실행
- YOLO 학습/탐지 실행
- 맵 저장

### c_cpp_properties.json
C++ IntelliSense 설정:
- ROS2 include 경로
- C++17 표준 설정
- 컴파일러 설정

## 🚀 시작하기

### 1. 권장 확장 프로그램 설치

VSCode에서 프로젝트를 열면 자동으로 권장 확장 프로그램 설치 알림이 표시됩니다.
또는 `Ctrl+Shift+P` → "Extensions: Show Recommended Extensions" 선택

### 2. Python 가상환경 설정

YOLO 프로젝트를 위한 가상환경:
```bash
cd yolo_proj
python3 -m venv volov5
source volov5/bin/activate
pip install -r requirements.txt
```

### 3. ROS2 환경 설정

터미널에서 자동으로 ROS2 환경이 로드됩니다. 수동으로 로드하려면:
```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
```

## 🎯 주요 기능 사용법

### 작업(Tasks) 실행
`Ctrl+Shift+P` → "Tasks: Run Task" → 원하는 작업 선택

자주 사용하는 작업:
- **TurtleBot3: Launch Gazebo World** - Gazebo 시뮬레이션 시작
- **TurtleBot3: Launch Navigation2** - Navigation2 실행
- **YOLO: Train Model** - YOLO 모델 학습
- **YOLO: Detect (Webcam)** - 웹캠으로 실시간 탐지

### 디버깅 시작
`F5` 키를 누르거나 디버그 패널에서 원하는 설정 선택:
- **Python: YOLO Train** - 학습 과정 디버깅
- **Python: YOLO Detect (Webcam)** - 탐지 과정 디버깅

### 터미널에서 ROS2 명령어 실행
VSCode 터미널을 열면 자동으로 ROS2 환경 변수가 설정됩니다:
```bash
ros2 topic list
ros2 node list
ros2 run rviz2 rviz2
```

## 🔧 사용자 정의

### Python 인터프리터 변경
`Ctrl+Shift+P` → "Python: Select Interpreter"

### 설정 수정
`settings.json`의 경로나 설정값을 프로젝트에 맞게 수정 가능

### 새 작업 추가
`tasks.json`에 새로운 작업을 추가하여 자주 사용하는 명령어를 단축키로 실행

## 📝 유용한 단축키

- `Ctrl+Shift+B`: 기본 빌드 작업 실행
- `F5`: 디버깅 시작
- `Ctrl+Shift+P`: 명령 팔레트
- `Ctrl+` `: 터미널 토글
- `Ctrl+K Ctrl+O`: 폴더 열기
- `Ctrl+P`: 파일 빠른 열기

## 🐛 문제 해결

### Python 인터프리터를 찾을 수 없음
가상환경 경로 확인:
```bash
which python3
# settings.json의 python.defaultInterpreterPath 업데이트
```

### ROS2 명령어를 찾을 수 없음
ROS2 환경 소싱:
```bash
source /opt/ros/humble/setup.bash
```

### C++ IntelliSense가 작동하지 않음
워크스페이스 빌드 후 `compile_commands.json` 생성 확인:
```bash
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## 📚 추가 정보

- [VSCode 공식 문서](https://code.visualstudio.com/docs)
- [ROS2 VSCode 확장](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros)
- [Python VSCode 확장](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

## 📧 문의

프로젝트 관련 문의: moojuengll2l@gmail.com
