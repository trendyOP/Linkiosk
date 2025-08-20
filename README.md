# 디지털 접근성 문제 해결을 위한 고령자 행동 모방 AI

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Models V2](https://huggingface.co/microsoft/OmniParser-v2.0)]

## 🎯 프로젝트 개요

**디지털 접근성 문제 해결을 위한 고령자 행동 모방 AI**는 고령자의 디지털 디바이드 문제를 해결하기 위해 개발된 혁신적인 AI 시스템입니다.

### 주요 기능
- **화면 파싱 및 요소 인식**: OmniParser V2를 통한 GUI 요소 자동 감지
- **고령자 시선 추적 모방**: 9×9 그리드 기반 시야 모델링
- **인지적 특성 모방**: 나이에 따른 반응 지연, 확실성 저하 모델링
- **키오스크 인터페이스**: React 기반 웹 키오스크 시스템
- **실시간 상호작용**: 실제 마우스 제어 및 화면 캡처

### 특장점
- **과학적 근거 기반**: 의학적 연구 데이터 기반 모델링
- **강화학습 기반 학습**: PPO 알고리즘을 통한 안정적인 정책 학습
- **실용적 적용**: 실제 키오스크 환경에서의 동작 검증

## 🚀 설치 및 실행 가이드

### 1단계: Python 환경 설정

#### 1.1 Python 설치
```bash
# Python 3.12 설치 (권장)
# https://www.python.org/downloads/ 에서 다운로드
# 설치 시 "Add Python to PATH" 체크 필수
```

#### 1.2 가상환경 생성
```bash
# 프로젝트 디렉토리로 이동
cd /c/Linkiosk-ppo_v2

# 가상환경 생성
python -m venv omni_env

# 가상환경 활성화 (Windows)
omni_env\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
# source omni_env/bin/activate
```

### 2단계: 필수 라이브러리 설치

#### 2.1 기본 의존성 설치
```bash
# requirements.txt 설치
pip install -r requirements.txt
```

#### 2.2 추가 필수 라이브러리
```bash
# 추가로 필요한 라이브러리들
pip install pygame
pip install mss
pip install pyautogui
pip install tkinter  # Python 기본 포함이지만 확인
pip install difflib  # Python 기본 포함
```

#### 2.3 CUDA 지원 PyTorch 설치 (GPU 사용 시)
```bash
# CUDA 11.8 지원 PyTorch (GPU 사용)
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# CPU 전용 PyTorch (GPU 없을 경우)
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```

### 3단계: 모델 파일 다운로드

#### 3.1 Hugging Face CLI 설치 (필요한 경우)
```bash
# Hugging Face CLI가 설치되어 있지 않다면
pip install huggingface_hub

# 또는
pip install --upgrade huggingface_hub
```

#### 3.2 OmniParser V2 모델 가중치 다운로드
```bash
# weights 디렉토리 생성
mkdir -p weights

# 방법 1: 자동 다운로드 (권장)
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do 
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

# 폴더명 변경 (V2에서는 icon_caption_florence 사용)
mv weights/icon_caption weights/icon_caption_florence

# 방법 2: 수동 다운로드 (자동 다운로드 실패 시)
# https://huggingface.co/microsoft/OmniParser-v2.0 에서 직접 다운로드
# - icon_detect/train_args.yaml
# - icon_detect/model.pt (39MB)
# - icon_detect/model.yaml
# - icon_caption/config.json
# - icon_caption/generation_config.json
# - icon_caption/model.safetensors (1.0GB)
```

#### 3.3 다운로드 확인
```bash
# 다운로드된 파일 확인
ls -la weights/icon_detect/
ls -la weights/icon_caption_florence/

# 예상 결과:
# weights/icon_detect/
# ├── model.pt (39MB)
# ├── model.yaml (1.1KB)
# └── train_args.yaml (1.7KB)
#
# weights/icon_caption_florence/
# ├── config.json (5.6KB)
# ├── generation_config.json (292B)
# └── model.safetensors (1.0GB)
```

#### 3.4 PPO 모델 파일 준비
```bash
# PPO 모델 파일이 없다면 생성 필요
# gaze_ppo_v9.pt 파일이 omniparser/ 디렉토리에 있어야 함

# 기존 PPO 모델이 있는지 확인
ls -la omniparser/gaze_ppo_v9.pt

# 없다면 다음 중 하나 선택:
# 1. 기존 훈련된 모델 파일 복사
# 2. 새로 훈련 (run_omniparser_with_ppo_v9.py 실행)
# 3. 샘플 모델 다운로드 (제공되는 경우)
```

#### 3.5 추가 모델 파일 (선택사항)
```bash
# icon_caption_blip 모델 (V1.5 호환성용)
# 이미 있다면 그대로 사용, 없다면 다운로드
if [ ! -d "weights/icon_caption_blip" ]; then
    echo "icon_caption_blip 모델이 없습니다. 필요시 다운로드하세요."
    echo "https://huggingface.co/microsoft/OmniParser/tree/main/icon_caption_blip2"
fi
```

#### 3.6 다운로드 문제 해결
```bash
# 네트워크 오류 시 재시도
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir weights --resume-download

# 캐시 삭제 후 재다운로드
rm -rf weights/.cache
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir weights

# 특정 파일만 다운로드
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors --local-dir weights
```

### 4단계: 설정 파일 준비

#### 4.1 config.ini 파일 확인
```ini
# omniparser/config.ini 파일이 다음 내용을 포함하는지 확인
[User]
type = elder
vision_grid = 9

[Menu]
queue = 매장식사, 아메리카노, HOT, 주문담기, 더담기, 베이커리, 햄치즈, 주문담기, 결제하기, 확인, 신용카드, 대기, 아니오
```

#### 4.2 필요한 파일들 확인
```bash
# 다음 파일들이 존재하는지 확인
ls -la screen5.png                    # 테스트용 스크린샷
ls -la click.wav                      # 클릭 사운드 파일
ls -la omniparser/gaze_ppo_v9.pt      # PPO 모델 파일
```

### 5단계: 디렉토리 구조 확인

#### 5.1 필수 디렉토리 구조
```
Linkiosk-ppo_v2/
├── omniparser/
│   ├── gaze_ppo_test_v9_2.py        # 실행할 파일
│   ├── run_omniparser_with_ppo_v9.py # 의존 파일
│   ├── config.ini                    # 설정 파일
│   ├── gaze_ppo_v9.pt               # PPO 모델
│   └── text_normalizer.py           # 텍스트 정규화
├── utils/
│   └── utils.py                     # 유틸리티 함수
├── weights/
│   ├── icon_detect/
│   └── icon_caption_florence/
├── screen5.png                      # 테스트 이미지
├── click.wav                        # 사운드 파일
└── requirements.txt                 # 의존성 목록
```

### 6단계: 실행 전 확인사항

#### 6.1 환경 변수 설정
```bash
# Windows PowerShell에서
$env:APPRAISAL_AGE = "70"
$env:APPRAISAL_PROPRIO_ERR_CM = "4.0"
$env:APPRAISAL_MODE = "RSv1"

# 또는 .env 파일 생성
echo "APPRAISAL_AGE=70" > .env
echo "APPRAISAL_PROPRIO_ERR_CM=4.0" >> .env
echo "APPRAISAL_MODE=RSv1" >> .env
```

#### 6.2 GPU 확인 (선택사항)
```python
# Python에서 GPU 확인
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name()}")
```

### 7단계: 실행

#### 7.1 기본 실행
```bash
# 프로젝트 루트 디렉토리에서
cd /c/Linkiosk-ppo_v2

# 가상환경 활성화
omni_env\Scripts\activate

# 실행
python omniparser/gaze_ppo_test_v9_2.py
```

#### 7.2 디버그 모드 실행
```bash
# 디버그 정보와 함께 실행
python -u omniparser/gaze_ppo_test_v9_2.py 2>&1 | tee debug.log
```

## 🔧 문제 해결

### 일반적인 오류 해결

#### ImportError: No module named 'utils'
```bash
# PYTHONPATH 설정
set PYTHONPATH=%PYTHONPATH%;%CD%
# 또는
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### CUDA out of memory
```bash
# GPU 메모리 부족 시 CPU 사용
export CUDA_VISIBLE_DEVICES=""
```

#### PaddleOCR 초기화 실패
```bash
# PaddleOCR 재설치
pip uninstall paddlepaddle paddleocr
pip install paddlepaddle-gpu==2.6.2 paddleocr==2.7.0.3
```

#### 파일 누락 시 대체 방법
```bash
# screen5.png가 없을 경우
# 아무 스크린샷이나 screen5.png로 복사

# click.wav가 없을 경우
# 아무 .wav 파일이나 click.wav로 복사하거나
# 코드에서 사운드 관련 부분 주석 처리
```

### 성공적인 실행 확인

정상 실행 시 다음과 같은 출력을 볼 수 있습니다:
```
[DPI] SetProcessDPIAware enabled - pyautogui가 물리 픽셀 좌표 사용
[SUCCESS] PaddleOCR 초기화 완료
[SUCCESS] torch import 완료
[SUCCESS] run_omniparser_with_ppo_v9 import 완료
[CONFIG] User type: elder
[CONFIG] Menu queue: ['매장식사', '아메리카노', 'HOT', ...]
== TASK 1: ['매장식사', '아메리카노', 'HOT', ...]
```

## 📚 참고 자료

### 논문 및 기술 문서
- [OmniParser for Pure Vision Based GUI Agent](https://arxiv.org/abs/2408.00203)
- [Project Page](https://microsoft.github.io/OmniParser/)
- [V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)

### 모델 다운로드
- [OmniParser V2 Models](https://huggingface.co/microsoft/OmniParser-v2.0)
- [OmniParser V1.5 Models](https://huggingface.co/microsoft/OmniParser)

## 📄 라이선스

모델 체크포인트의 라이선스:
- **icon_detect 모델**: AGPL 라이선스 (YOLO 모델 상속)
- **icon_caption_florence**: MIT 라이선스
- **icon_caption_blip**: MIT 라이선스

각 모델의 라이선스 파일은 [Hugging Face 모델 페이지](https://huggingface.co/microsoft/OmniParser)에서 확인할 수 있습니다.

## 📞 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해주세요.

---


# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>
<!-- <a href="https://trendshift.io/repositories/12975" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12975" alt="microsoft%2FOmniParser | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Models V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[Models V1.5](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Space Demo](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface. 

## News
- [2025/3] We support local logging of trajecotry so that you can use OmniParser+OmniTool to build training data pipeline for your favorate agent in your domain. [Documentation WIP]
- [2025/3] We are gradually adding multi agents orchstration and improving user interface in OmniTool for better experience.
- [2025/2] We release OmniParser V2 [checkpoints](https://huggingface.co/microsoft/OmniParser-v2.0). [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] We introduce OmniTool: Control a Windows 11 VM with OmniParser + your vision model of choice. OmniTool supports out of the box the following large language models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use. [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 is coming. We achieve new state of the art results 39.5% on the new grounding benchmark [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) with OmniParser v2 (will be released soon)! Read more details [here](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md).
- [2024/11] We release an updated version, OmniParser V1.5 which features 1) more fine grained/small icon detection, 2) prediction of whether each screen element is interactable or not. Examples in the demo.ipynb. 
- [2024/10] OmniParser was the #1 trending model on huggingface model hub (starting 10/29/2024). 
- [2024/10] Feel free to checkout our demo on [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)! (stay tuned for OmniParser + Claude Computer Use)
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Hugginface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)! 

## Install 
First clone the repo, and then install environment:
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Ensure you have the V2 weights downloaded in weights folder (ensure caption weights folder is called icon_caption_florence). If not download them with:
```
   # download the model checkpoints to local directory OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [deprecated]
Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

For v1: 
convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py

For v1.5: 
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed. 
``` -->

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Gradio Demo
To run gradio demo, simply run:
```python
python gradio_demo.py
```

## Model Weights License
For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

## 📚 Citation
Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing our work:
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
