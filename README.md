# PubMedQA: Full Fine-tuning vs. LoRA

Qwen2.5-1.5B-Instruct를 PubMedQA PQA-L에 적용하여 Inference Only,
Full Fine-tuning, LoRA를 비교한 실험 코드입니다. 보고서에 사용한 모델은
**LoRA v5**와 **Full Fine-tuning v3**이며, 학습·교차검증·최종 평가 모두
문맥의 섹션 라벨을 입력에 포함하지 않습니다.

## 실험 구성

- 기본 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: PubMedQA PQA-L 1,000개
- 모델 선택 데이터: 공식 10-fold의 500개 샘플
- 최종 평가 데이터: 공식 test set 500개
- 분류 라벨: `yes`, `no`, `maybe`
- 클래스 손실 가중치: `1.0`, `1.0`, `1.5`
- LoRA: rank 16, alpha 32, dropout 0.05
- 시드: 42

## 파일 구성

```text
src/
├── 01_download_dataset.py       # 공식 데이터 다운로드 및 split 검증
├── 02_inference_only.py         # 기본 모델 최종 평가
├── 03_finetune.py               # 10-fold CV 및 전체 데이터 재학습
├── 04_evaluate_finetuned.py     # LoRA v5 / Full v3 최종 평가
├── 05_visualize_results.py      # 보고서 그림 1, 2, 3, 10 생성
├── training_setup.py            # 모델·LoRA·Trainer 설정
├── training_utils.py            # 데이터 전처리와 collator
├── training_recorder.py         # 학습 결과 기록
└── utils.py                     # 공통 프롬프트·평가·JSON 함수
```

## 설치

Python 3.11과 CUDA GPU 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행 순서

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python src/01_download_dataset.py
python src/02_inference_only.py

python src/03_finetune.py --method lora
python src/03_finetune.py --method full

python src/04_evaluate_finetuned.py --method lora
python src/04_evaluate_finetuned.py --method full

python src/05_visualize_results.py
```

`03_finetune.py`는 각 방법에 대해 10-fold 학습 후 500개 전체 학습
샘플로 최종 모델을 다시 학습하므로 실행 시간이 오래 걸리고 저장 공간을
많이 사용합니다. 결과와 체크포인트는 `outputs/` 아래에 저장됩니다.
최종 평가 JSON은 `outputs/qwen2.5-1.5b/evaluation/`, 보고서 그림은
`outputs/qwen2.5-1.5b/model_comparison_visualizations/`에 저장됩니다.

## 주요 결과

| 방법 | 최종 Accuracy | 최종 Macro F1 |
|---|---:|---:|
| Inference Only | 0.670 | 0.446 |
| LoRA | 0.746 | 0.533 |
| Full Fine-tuning | 0.750 | 0.553 |

| 방법 | 10-fold Accuracy | 10-fold Macro F1 |
|---|---:|---:|
| LoRA | 0.742 ± 0.055 | 0.558 ± 0.076 |
| Full Fine-tuning | 0.774 ± 0.043 | 0.586 ± 0.070 |

## 재현성 참고

- 모든 방법은 동일한 시스템 프롬프트와 섹션 라벨 없는 입력을 사용합니다.
- 평가는 생성 문자열 파싱 대신 첫 답변 위치의 `yes/no/maybe` 토큰 logit 중
  가장 큰 값을 선택합니다.
- 대용량 체크포인트, 실행 로그 및 원시 결과는 `.gitignore`로 제외됩니다.
- PubMedQA 데이터는 실행 시 [공식 저장소](https://github.com/pubmedqa/pubmedqa)에서
  내려받습니다.
