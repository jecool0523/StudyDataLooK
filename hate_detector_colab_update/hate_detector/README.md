# KoMultiText + KcELECTRA Binary Hate Detector

디시인사이드 글을 `정상 / 유해(욕설/혐오/공격적 표현)`로 먼저 거르는 1단계 탐지기입니다.

## 설치

```powershell
python -m pip install -r requirements-hate-detector.txt
```

Codex 번들 Python을 쓸 때는 아래처럼 실행할 수 있습니다.

```powershell
& 'C:\Users\seocheon\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -r requirements-hate-detector.txt
```

## 학습

KoMultiText의 `profanity`, `gender`, `politics`, `nation`, `race`, `region`, `generation`, `social_hierarchy`, `appearance`, `others` 중 하나라도 1이면 `harmful=1`로 변환합니다.

```powershell
python -m hate_detector.train_binary --output-dir models/kc-electra-komultitext-binary --epochs 3
```

CPU 환경에서 먼저 학습 코드만 빠르게 확인하려면:

```powershell
python -m hate_detector.train_binary --output-dir models/smoke-kc-electra --epochs 0.02 --max-train-samples 64 --max-eval-samples 64
```

1 epoch 학습 후 같은 output directory에서 총 3 epoch까지 이어서 학습하려면:

```powershell
python -m hate_detector.train_binary --output-dir models/kc-electra-komultitext-binary --epochs 3 --resume-from-checkpoint
```

## 임계값 튜닝

recall을 우선으로 잡기 위해 검증셋 예측 확률에서 임계값을 고릅니다. 기본값은 `recall >= 0.90`을 만족하는 후보 중 F1이 가장 높은 임계값입니다.

```powershell
python -m hate_detector.tune_threshold --model-dir models/kc-electra-komultitext-binary --min-recall 0.90
```

결과는 `models/kc-electra-komultitext-binary/threshold.json`에 저장됩니다.

## 디시 CSV 추론

금칙어 필터와 모델 점수를 결합합니다. 최종 점수는 `max(model_score, keyword_score)`입니다.

```powershell
python -m hate_detector.predict_csv --model-dir models/kc-electra-komultitext-binary --input crawler/di.csv --output "분석 데이터/predicted_di_hate.csv"
```

## 디시 CSV 폴더 일괄 추론

`crawler/dc/` 아래 모든 CSV를 재귀적으로 분석합니다.

```powershell
python -m hate_detector.predict_dir --model-dir models/kc-electra-komultitext-binary --input-dir crawler/dc --output-dir "분석 데이터/hate_predictions"
```

`schools_20p/all_schools.csv`처럼 개별 CSV의 합본을 중복 집계에서 빼고 싶으면:

```powershell
python -m hate_detector.predict_dir --model-dir models/kc-electra-komultitext-binary --input-dir crawler/dc --output-dir "분석 데이터/hate_predictions" --exclude-name all_schools.csv
```

출력:

- `분석 데이터/hate_predictions/*_hate.csv`: 원본 파일별 예측 결과
- `분석 데이터/hate_predictions/summary.csv`: 파일별 유해 글 수/비율 요약
- `분석 데이터/hate_predictions/run_config.json`: 사용한 모델/임계값 기록

## 학습 전 금칙어 사전 점검

모델 학습 전에도 `crawler/dc/` 전체에서 금칙어 기반 후보를 빠르게 확인할 수 있습니다.

```powershell
python -m hate_detector.keyword_scan_dir --input-dir crawler/dc --output-dir "분석 데이터/keyword_scan"
```

## 출력 컬럼

- `harmful_score`: 모델 점수와 금칙어 점수를 합친 최종 위험도
- `model_score`: KcELECTRA binary classifier의 유해 확률
- `keyword_score`: 금칙어 필터 점수
- `is_harmful`: 최종 유해 판정
- `matched_terms`: 매칭된 금칙어 목록
