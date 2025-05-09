YOLOv5 기반 3클래스 객체 탐지 프로젝트 (person, bird, crack)

Author: mj @ mj/yolo_projFrameworks: PyTorch, YOLOv5, OpenCV, Flask (예정)Platform: Windows 11 + VSCode + Anaconda + Python 3.10 + venv(volov5)

✨ 프로젝트 개요

이 프로젝트는 YOLOv5를 이용하여 총 3가지 객체(person, bird, crack)를 학습시키고 실시간 웹캠 또는 이미지/영상 파일을 통해 객체를 탐지하는 데 목적이 있다.

본 모델은 ROS2 Humble과 연동하여 스마트 감시/자율주행 시스템으로 확장될 예정이며, 실시간 감지를 위한 웹서버(Flask)도 구현될 계획이다.

📂 디렉토리 구조

yolo_proj/
├── yolov5/                    # YOLOv5 전체 프로젝트
│   ├── datasets/             # 학습용 데이터셋
│   │   └── 3class/
│   │       ├── train/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       └── valid/
│   │           ├── images/
│   │           └── labels/
│   ├── runs/train/gassib/     # 학습 결과
│   │   ├── results.png
│   │   └── weights/best.pt
│   ├── detect.py              # 실시간 감지 실행 파일
│   ├── train.py               # 학습 실행 파일
│   ├── 3class.yaml            # 데이터셋 정보 정의 파일
│   └── datasets/hyps/         # 하이퍼파라미터 설정
│
└── volov5/                   # venv 가상환경 폴더

⚙️ 가상환경 실행 방법 (venv: volov5)

# 가상환경 진입 (Windows 기준)
.olov5\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt

⚖️ 학습 명령어

python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data datasets/3class/data.yaml \
  --weights yolov5s.pt \
  --name gassib \
  --hyp datasets/hyps/hyp.scratch-high.yaml

--img: 입력 이미지 크기

--batch: 배치 크기

--epochs: 반복 횟수

--data: 클래스 및 경로 정의된 YAML

--weights: 사전학습 모델 또는 '' (scratch)

--hyp: 하이퍼파라미터 설정 파일

--name: 학습 세션 이름 (결과는 runs/train/{name})

🔍 감지 실행 명령어

📹 실시간 웹캠 감지

python detect.py \
  --weights runs/train/gassib/weights/best.pt \
  --source 0 \
  --img 640 \
  --data datasets/3class/data.yaml

🖼️ 이미지 폴더 감지

python detect.py \
  --weights runs/train/gassib/weights/best.pt \
  --source datasets/3class/valid/images \
  --img 640 \
  --data datasets/3class/data.yaml

🎞️ 영상 감지

python detect.py \
  --weights runs/train/gassib/weights/best.pt \
  --source path/to/video.mp4 \
  --img 640 \
  --data datasets/3class/data.yaml

💾 결과 저장 위치

결과 이미지/영상: runs/detect/exp*/

📄 data.yaml 예시 (datasets/3class/data.yaml)

train: datasets/3class/train/images
val: datasets/3class/valid/images

nc: 3
names: ['bird', 'crack', 'person']

⚙️ 추후 계획

Flask 웹서버 연동하여 웹 브라우저로 실시간 감지 결과 송출 예정

ROS2 Humble + SLAM + Nav2 + LLM 제어 통합

객체 탐지 기반 자율주행 로봇에 탑재

📚 참고 리포지터리

Ultralytics YOLOv5

PyTorch

OpenCV

Roboflow

✨ 기타 주의사항

data.yaml, 라벨 .txt 는 반드시 클래스 ID 0~2 범위로 구성되어야 함

labels.cache 관련 오류가 있으면 해당 파일을 삭제하고 재실행

.pt 파일은 runs/train/{세션명}/weights/best.pt 기준 사용

문의: moojuengll2l@gmail.com / 연구실용본 프로젝트는 학부 산학연계프로젝트 / ROS2 자율주행 기반 감시시스템 실험용


이 부분을 readme파일로 만들어줘 다운받아서 github에 올릴거야