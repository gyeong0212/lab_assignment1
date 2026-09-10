Assignment 1: PubMedQA Fine-tuning
1. 과제 개요
Qwen2.5-1.5B-Instruct 모델을 PubMedQA 데이터셋에 적용하여 Inference Only, Full Fine-tuning, LoRA 방식의 성능과 효율을 비교하는 과제입니다.
2. 모델 및 데이터셋
- Base Model: Qwen/Qwen2.5-1.5B-Instruct
- Dataset: PubMedQA PQA-L
- Labels: yes, no, maybe
- Evaluation Metrics: Accuracy, Macro F1
- Validation Method: 10-fold Cross-validation
전체 데이터 1,000개 중 500개는 학습 및 모델 선택에 사용하고, 나머지 500개는 최종 평가에 사용합니다. 각 fold는 학습 데이터 450개와 검증 데이터 50개로 구성됩니다.
3. 파일 구성
src/
├── 01_download_dataset.py
├── 02_inference_only.py
├── 03_finetune.py
├── 04_evaluate_finetuned.py
├── training_recorder.py
├── training_setup.py
├── training_utils.py
├── utils.py
└── wandb_utils.py
- 01_download_dataset.py: PubMedQA 데이터셋 다운로드 및 준비
- 02_inference_only.py: 미세조정하지 않은 기반 모델의 zero-shot 평가
- 03_finetune.py: Full Fine-tuning 또는 LoRA 학습
- 04_evaluate_finetuned.py: 저장된 미세조정 모델의 성능 평가
- training_recorder.py: 학습 결과와 리소스 사용량 기록
- training_setup.py: 모델 및 학습 설정 관리
- training_utils.py: 데이터 로딩, 토큰화 및 학습 보조 기능
- utils.py: 프롬프트 생성, 출력 파싱 및 평가 지표 계산
- wandb_utils.py: Weights & Biases 실험 기록 기능
4. 실행 환경
- Python 3.11
- CUDA 지원 GPU 권장
- PyTorch
- Transformers
- Datasets
- PEFT
- scikit-learn
- Weights & Biases
