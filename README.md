# CUDA-accelerated SLAM ICP

```text
src_cpu/              CPU ICP 실행 및 단계별 계산 함수
src_cuda/             CUDA ICP 실행 및 GPU 커널
src_common/           공통 CLI 진입점
include/icp/          공통 자료형과 함수 선언
scripts/              데이터 준비, 평가, 벤치마크 도구
tests/               C++ 및 Python 테스트
data/synthetic_small/ 정답 변환이 포함된 작은 합성 데이터
build-cpu/            CPU 빌드 결과 (Git 제외)
build-gpu/            CUDA 빌드 결과 (Git 제외)
results/              실행·검증 결과 (Git 제외)
```

`CMakeLists.txt`: C++/CUDA 빌드 설정 · `requirements.txt`: Python 의존성
