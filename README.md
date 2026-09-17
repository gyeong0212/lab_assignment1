# PubMedQA: Inference Only, Full Fine-tuning, LoRA

Qwen2.5-1.5B-Instruct와 PubMedQA PQA-L의 공식 분할로 세 방법을 비교하는 코드입니다.
모든 방법은 문맥의 섹션 라벨을 포함한 동일한 입력을 사용합니다.

## 실험 절차

- 공식 test set 500개를 최종 평가에만 사용합니다.
- 나머지 500개를 공식 10개 fold로 나누어, 방법별로 각 fold의 450개에서 기본 모델부터 3 epoch 학습합니다.
- 각 fold에서 검증 Macro F1이 가장 높은 epoch의 모델을 저장합니다. Full Fine-tuning은 전체 가중치를, LoRA는 rank 16, alpha 32, dropout 0.05의 어댑터를 학습합니다.
- 두 방법 모두 `yes/no/maybe` 첫 답변 토큰에만 손실을 계산하며, 손실 가중치는 `1/1/1.5`입니다. 학습률은 Full Fine-tuning `2e-5`, LoRA `1e-4`이고, 실질적인 batch size는 8입니다.
- 각 fold의 검증 예측을 합친 OOF 결과로 출력 보정값을 비교합니다. Accuracy와 클래스별 F1을 유지하는 후보 중 Macro F1이 가장 높은 보정을 선택합니다. 적합한 후보가 없으면 보정을 적용하지 않습니다.
- 최종 평가에서는 각 방법의 fold 모델 10개를 test set의 각 샘플에 적용하고, 클래스별 logit을 평균한 뒤 OOF에서 결정한 보정을 반영합니다. 전체 500개로 별도 재학습하지 않습니다.

## 실행

CUDA GPU와 Python 3.11 환경에서 의존성을 설치한 뒤, 프로젝트 루트에서 실행합니다.

```bash
pip install -r requirements.txt
python src/01_download_dataset.py
python src/02_inference_only.py
python src/03_finetune.py --method lora
python src/03_finetune.py --method full
python src/04_evaluate_finetuned.py --method lora
python src/04_evaluate_finetuned.py --method full
```

`03_finetune.py`는 fold별 모델, OOF 점수, 교차검증 요약, 보정 선택 결과를
`outputs/qwen2.5-1.5b/` 아래에 저장합니다. `04_evaluate_finetuned.py`는
저장된 fold 모델과 OOF 점수를 사용하여 최종 앙상블 예측을 저장합니다.
Full Fine-tuning의 저장된 OOF 점수에서는 `maybe` logit `-0.1`이 선택되고,
LoRA에서는 출력 보정을 적용하지 않습니다.

## 보고서 최종 평가 결과

| 방법 | Accuracy | Macro F1 |
|---|---:|---:|
| Inference Only | 0.676 | 0.454 |
| LoRA | 0.750 | 0.547 |
| Full Fine-tuning | 0.754 | 0.574 |

LoRA의 모델 저장 용량은 어댑터만 포함하며, 추론에는 기본 모델도 필요합니다.
