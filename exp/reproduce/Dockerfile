# ARM64 CPU-only reproducibility 실험용 Dockerfile

FROM ubuntu:22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y python3.10 python3-pip git

# pip 업그레이드
RUN python3.10 -m pip install --upgrade pip

# ARM64용 PyTorch (CPU-only) 설치
RUN python3.10 -m pip install torch==2.2.0 torchvision==0.17.0 numpy==1.24.4

# 작업 디렉토리 생성
WORKDIR /workspace
