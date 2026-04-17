# 📸 Canon AI Project: Intelligent Quality Inspection System

**YOLO, CNN, OCR 기술을 결합한 지능형 이미지 분석 및 실시간 품질 검사 플랫폼**

본 프로젝트는 제조 및 검수 공정에서 발생할 수 있는 결함을 AI 모델을 통해 실시간으로 탐지하고 분류하는 통합 시스템입니다. FastAPI 기반의 고성능 모델 서빙 인프라와 Next.js 기반의 직관적인 관리 대시보드를 제공합니다.

---

## 🚀 핵심 기능 (Key Features)

### 1. Multi-Model AI 분석 파이프라인
*   **Object Detection (YOLO)**: 이미지 내 핵심 부품 및 객체 위치 탐지.
*   **Classification (CNN)**: 탐지된 객체의 상태(Pass/Fail)를 정밀 분류.
*   **Text Recognition (OCR)**: 제품 내 각인된 텍스트 및 시리얼 번호를 추출하여 데이터화.

### 2. 실시간 모니터링 & 배치 분석
*   **Live Stream Analysis**: 연결된 카메라를 통해 실시간으로 프레임을 분석하고 즉각적인 피드백 제공.
*   **Batch Processing**: 다량의 이미지를 업로드하여 일괄적으로 분석하고 분석 리포트 생성.

### 3. 데이터 시각화 및 관리
*   **Analysis Dashboard**: 분석 결과(합격/불합격), 신뢰도(Confidence), 실패 사유 등을 그리드 뷰로 시각화.
*   **Statistical Analytics**: 전체 공정의 합격률 및 오류 유형별 통계 대시보드 제공.
*   **Report Export**: 분석 완료 데이터를 CSV 형식으로 추출하여 데이터 자산화 지원.

---
## 모델 테스트

<img width="600" height="875" alt="image" src="https://github.com/user-attachments/assets/a96072c9-a7ab-4c25-a033-5cb7adf5449e" />
<img width="600" height="720" alt="1-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/83deac98-2f8d-41da-9b13-c2c7eec9e3a5" />
<img width="600" height="450" alt="2-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/ec300cd5-fa95-48fe-8ccf-8fd834415d0d" />

---
## 🛠 기술 스택 (Tech Stack)

### Backend & AI
*   **Framework**: FastAPI (Python 3.11+)
*   **AI Models**: PyTorch (YOLO, Custom CNN Classifier), Tesseract/OCR
*   **Database**: SQLite (SQLAlchemy ORM)
*   **Serving**: Uvicorn

### Frontend
*   **Framework**: Next.js 14 (TypeScript)
*   **Styling**: Tailwind CSS, Shadcn/UI
*   **Icons**: Lucide-React
*   **State Management**: React Hooks

---

## 🏗 시스템 아키텍처 (Architecture)

1.  **Client**: 사용자 이미지 업로드 또는 실시간 카메라 스트리밍 요청 (Next.js)
2.  **API Server**: 이미지 전처리 및 AI 모델 추론 파이프라인 실행 (FastAPI)
3.  **Model Inference**: YOLO(탐지) -> CNN(분류) -> OCR(인식) 순차 처리
4.  **Database**: 분석 결과 및 통계 데이터 영구 저장 (SQLite)
5.  **Response**: 최종 판단 결과 및 시각화 데이터를 대시보드에 반환

---


## PyTorch DLL 로드 오류 해결 보고서 (WinError 1114)

PyTorch 기반 FastAPI 서버 실행 중 발생했던 `OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다. Error loading "C:\...\torch\lib\c10.dll" or one of its dependencies` 오류에 대한 문제 진단 및 해결 과정을 보고서 형식으로 정리했습니다.

-----

### 1\. 📌 문제 개요

| 구분 | 내용 |
| :--- | :--- |
| **발생 오류** | `OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다.` |
| **원인** | PyTorch의 핵심 라이브러리인 `c10.dll`이 로드될 때, 해당 DLL이 의존하는 **시스템 종속성(Visual C++ 런타임 등)** 파일을 찾지 못하거나, **잘못된 순서로 로드**되어 초기화에 실패함. |
| **근본 원인** | 가상 환경의 내부 DLL 경로가 Windows의 **시스템/사용자 환경 변수(`Path`)에 수동으로 등록**되어 PyTorch의 정상적인 로드 메커니즘을 방해함. |
| **영향** | FastAPI 서버(`main.py`) 실행 불가. |

-----

### 2\. 🔍 초기 진단 및 해결 시도

| 단계 | 시도 내용 | 결과 | 비고 |
| :--- | :--- | :--- | :--- |
| **1단계** | **Visual C++ Redistributable 설치/복구** | 실패 | DLL 종속성 파일 자체의 누락 문제는 아니었음을 확인. |
| **2단계** | PyTorch 및 관련 패키지 **클린 재설치** | 실패 | `torchvision` 버전 태그 오류 (`+cpu`) 등 설치 자체에 어려움 발생. |
| **3단계** | **`CUDA_VISIBLE_DEVICES=""` 환경 변수 설정** | 실패 | 오류 원인이 GPU/CUDA 로드 시도 때문만은 아님을 확인. |
| **4단계** | PyTorch 버전 **다운그레이드 시도** | 실패 | 설치 인덱스 문제로 `1.x` 버전 설치 불가. |

-----

### 3\. 🎯 최종 진단 및 문제 해결 (핵심)

최종 진단 결과, 오류는 Windows의 **DLL 검색 순서 문제**였으며, 다음 조치를 통해 해결되었습니다.

#### A. 환경 변수 경로 정리 (핵심 해결)

1.  **진단:** `Path` 환경 변수에 **가상 환경 내부의 DLL 경로**가 등록되어 있음을 확인했습니다.
      * **문제 경로:** `C:\GIthub\CanonProject_real\.venv\Lib\site-packages\torch\lib`
2.  **조치:** Windows **시스템/사용자 환경 변수(`Path`)** 목록에서 위 경로를 **즉시 삭제**했습니다.
3.  **효과:** PyTorch의 DLL 로드 시퀀스에 대한 외부 간섭이 제거되었고, `import torch` 시 가상 환경 내부의 정상적인 로드 절차가 작동하게 되었습니다.

#### B. PyTorch 클린 설치 재시도 및 버전 통일

환경 변수 정리 후, 호환성 확보를 위해 PyTorch를 올바른 명령어로 재설치했습니다.

1.  **기존 환경 정리:**
    ```bash
    deactivate
    pip uninstall torch torchvision torchaudio -y
    ```
2.  **호환 버전 설치:** (Python 3.11 환경에 맞는 2.x 버전대 중 안정적인 버전 선택)
    ```bash
    pip install torch==2.3.1+cpu torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
    ```

#### C. 서버 실행

1.  **경로 이동:**
    ```bash
    cd server
    ```
2.  **GPU 충돌 방지 환경 변수 설정:** (안전 확보 차원에서 항상 실행)
    ```bash
    set CUDA_VISIBLE_DEVICES=""
    ```
3.  **서버 실행:**
    ```bash
    python main.py
    ```

-----

### 4\. 📝 향후 참고 사항

1.  **DLL Path 관리:** **어떤 경우에도 가상 환경(`venv`) 내부의 경로는 Windows 시스템/사용자 환경 변수(`Path`)에 직접 등록하지 않습니다.** 가상 환경 활성화가 그 역할을 대신합니다.
2.  **PyTorch 설치:** `pip install` 시 `+cpu` 태그가 인식되지 않을 경우, 반드시 `--index-url https://download.pytorch.org/whl/cpu` 옵션을 사용하거나, `-f https://download.pytorch.org/whl/cpu/torch_stable.html` 옵션을 사용해야 합니다.
3.  **WinError 1114 발생 시 최우선 조치:**
      * Visual C++ Redistributable 복구/재설치
      * **환경 변수(`Path`)에서 가상 환경 관련 DLL 경로 즉시 제거**
