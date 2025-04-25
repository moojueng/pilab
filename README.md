# ROS2 Humble 기반 자율주행 및 LLM 제어 프로젝트

## 📌 프로젝트 개요

이 저장소는 **ROS2 Humble**을 기반으로 한 모바일 로봇의 자율주행 구현과, **LLM (Large Language Model)**을 이용한 자연어 기반 로봇 제어 실험을 정리한 프로젝트입니다. 향후 **로봇 매니퓰레이터** 제어와의 통합을 목표로 하고 있습니다.

---

## 📁 프로젝트 구성

### 1. `ros2_humble_nav2`
- **목적:** ROS2 Nav2 패키지를 활용하여 모바일 로봇의 자율주행 구현
- **주요 기능:**
  - SLAM을 통한 실시간 맵 생성
  - 목표 위치(Point Goal) 기반 자동 경로 생성 및 추종
  - 장애물 회피 및 costmap 기반의 동적 경로 재계산

### 2. `ros2_humble_nav2_LLM_control`
- **목적:** LLM을 이용한 자연어 기반 경로 지시 및 제어
- **시스템 개요:**
  - 사용자의 자연어 명령 → LLM 처리 → ROS2 명령어 변환
  - ex) "거실로 이동해" → `/navigate_to_pose` 토픽 발행
- **구현 방식:**
  - OpenAI API 또는 Fine-tuned LLM + Python 인터페이스
  - 텍스트 입력 또는 음성(STT) → 명령 추출 → ROS2 연동

---

## 🧠 향후 확장 계획

### 3. Manipulator 제어 시스템
- **목표:** 모바일 베이스 + 매니퓰레이터 통합 지능형 로봇 시스템
- **예정 기능:**
  - 시뮬레이션 환경(Gazebo, Mujoco)에서 6-DOF 이상 매니퓰레이터 제어
  - LLM + 비전 기반 작업 명령 수행 (ex: “책을 집어서 테이블 위에 올려줘”)
  - DDPM 등 확률 기반 경로 생성 알고리즘과의 연계

---

## 🔧 사용 환경

- **OS:** Ubuntu 22.04
- **ROS2:** Humble Hawksbill
- **시뮬레이터:** Gazebo / Mujoco (선택)
- **언어:** Python, C++
- **LLM:** OpenAI GPT / KoAlpaca 등 활용 가능

---

## 💡 연구 키워드

`ROS2` `Nav2` `LLM Control` `지능형 로봇` `자율주행` `Manipulator` `Gazebo` `딥러닝 제어` `Multimodal Robotics`

---

## 👨‍🔬 연구자 정보

- **소속:** 경북대학교 물·IT융합공학과
- **연구실:** 이상문 교수님 지능형 로봇 연구실
- **작성자:** 김무정 석사

---

## 🔗 참고 자료

- [ROS2 Navigation2 공식 문서](https://navigation.ros.org/)
- [OpenAI GPT API](https://platform.openai.com/docs/)
- [Franka Control with ROS2](https://github.com/frankaemika/franka_ros2)
- [Gazebo Simulation with ROS2](https://gazebosim.org/)

