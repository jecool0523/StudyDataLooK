# Colab 실행 가이드: Binary + 혐오 유형 분류

이 가이드는 Colab GPU에서 다음 흐름을 실행합니다.

1. KoMultiText + KcELECTRA binary 유해표현 탐지기 학습
2. binary 모델로 CSV에서 유해 데이터 추출
3. KoMultiText 유해 라벨만 사용해 혐오 유형 멀티라벨 모델 학습
4. 유해 데이터에 `primary_hate_type`, `matched_hate_types` 컬럼 추가

## 1. Colab 런타임 설정

Colab 상단 메뉴에서 `런타임 > 런타임 유형 변경 > 하드웨어 가속기 > GPU`를 선택합니다.

```python
import torch

print("cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. Google Drive 마운트

프로젝트 폴더를 Google Drive에 올린 뒤 Colab에서 마운트합니다. 아래 경로는 예시입니다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd "/content/drive/MyDrive/HEAR/code"
```

## 3. 패키지 설치

```python
!pip install -q -r requirements-hate-detector.txt
```

```python
import torch, transformers, datasets, sklearn

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("cuda", torch.cuda.is_available())
```

## 4. 코드 Smoke Test

전체 학습 전에 작은 샘플로 binary 모델 코드가 정상 실행되는지 확인합니다.

```python
!python -m hate_detector.train_binary \
  --output-dir models/smoke-kc-electra-binary \
  --epochs 0.01 \
  --max-train-samples 16 \
  --max-eval-samples 16 \
  --batch-size 4 \
  --max-length 64
```

유형 분류 모델 코드도 작은 샘플로 확인합니다.

```python
!python -m hate_detector.train_hate_type \
  --output-dir models/smoke-kc-electra-hate-type \
  --epochs 0.01 \
  --max-train-samples 16 \
  --max-eval-samples 16 \
  --batch-size 4 \
  --max-length 64
```

## 5. Binary 모델 학습

```python
!python -m hate_detector.train_binary \
  --output-dir models/kc-electra-komultitext-binary \
  --epochs 3 \
  --batch-size 16 \
  --max-length 160 \
  --harmful-class-weight 1.8
```

GPU 메모리가 부족하면 `--batch-size 8`로 낮춥니다.

중간에 끊긴 학습을 이어서 실행하려면:

```python
!python -m hate_detector.train_binary \
  --output-dir models/kc-electra-komultitext-binary \
  --epochs 3 \
  --batch-size 16 \
  --max-length 160 \
  --harmful-class-weight 1.8 \
  --resume-from-checkpoint
```

## 6. Binary Threshold 조정

유해 데이터를 놓치지 않는 쪽을 우선하려면 recall 기준으로 threshold를 조정합니다.

```python
!python -m hate_detector.tune_threshold \
  --model-dir models/kc-electra-komultitext-binary \
  --min-recall 0.90 \
  --batch-size 64 \
  --max-length 160
```

```python
import json

with open("models/kc-electra-komultitext-binary/threshold.json", encoding="utf-8") as f:
    print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
```

## 7. CSV 전체 Binary 예측

`crawler/dc/` 아래 CSV 전체를 분석합니다. `all_schools.csv`가 개별 학교 CSV의 합본이면 중복 집계를 피하기 위해 제외합니다.

```python
!python -m hate_detector.predict_dir \
  --model-dir models/kc-electra-komultitext-binary \
  --input-dir crawler/dc \
  --output-dir "분석 데이터/hate_predictions" \
  --exclude-name all_schools.csv \
  --batch-size 64 \
  --max-length 192
```

```python
import pandas as pd

binary_summary = pd.read_csv("분석 데이터/hate_predictions/summary.csv")
binary_summary
```

## 8. 혐오 유형 모델 학습

KoMultiText에서 혐오 유형이 하나라도 켜진 문장만 남기고 학습합니다. 한 문장에 여러 유형이 동시에 있을 수 있으므로 멀티라벨 분류로 학습합니다.

```python
!python -m hate_detector.train_hate_type \
  --output-dir models/kc-electra-komultitext-hate-type \
  --epochs 3 \
  --batch-size 16 \
  --max-length 160 \
  --threshold 0.5
```

GPU 메모리가 부족하면 `--batch-size 8`로 낮춥니다.

중간에 끊긴 학습을 이어서 실행하려면:

```python
!python -m hate_detector.train_hate_type \
  --output-dir models/kc-electra-komultitext-hate-type \
  --epochs 3 \
  --batch-size 16 \
  --max-length 160 \
  --threshold 0.5 \
  --resume-from-checkpoint
```

학습된 라벨 구성을 확인합니다.

```python
import json

with open("models/kc-electra-komultitext-hate-type/label_config.json", encoding="utf-8") as f:
    print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
```

## 9. Binary 유해 데이터에 유형 예측 추가

기본값은 `is_harmful=1`인 행에만 유형 모델을 적용합니다.

```python
!python -m hate_detector.predict_hate_type_dir \
  --type-model-dir models/kc-electra-komultitext-hate-type \
  --input-dir "분석 데이터/hate_predictions" \
  --output-dir "분석 데이터/hate_type_predictions" \
  --batch-size 64 \
  --max-length 192
```

모든 행에 유형 모델을 적용하고 싶으면 `--classify-all`을 추가합니다.

```python
!python -m hate_detector.predict_hate_type_dir \
  --type-model-dir models/kc-electra-komultitext-hate-type \
  --input-dir "분석 데이터/hate_predictions" \
  --output-dir "분석 데이터/hate_type_predictions_all" \
  --classify-all \
  --batch-size 64 \
  --max-length 192
```

## 10. 결과 확인

```python
import pandas as pd

type_summary = pd.read_csv("분석 데이터/hate_type_predictions/summary.csv")
type_summary
```

```python
from pathlib import Path
import pandas as pd

files = sorted(Path("분석 데이터/hate_type_predictions").glob("*_hate.csv"))
dfs = [pd.read_csv(path) for path in files]
all_typed = pd.concat(dfs, ignore_index=True)

cols = [
    "source_file",
    "제목",
    "내용",
    "harmful_score",
    "primary_hate_type",
    "matched_hate_types",
]
existing_cols = [col for col in cols if col in all_typed.columns]
all_typed.sort_values("harmful_score", ascending=False)[existing_cols].head(30)
```

유형별 총량을 확인합니다.

```python
type_cols = [col for col in all_typed.columns if col.startswith("hate_type_") and not col.endswith("_score")]
all_typed[type_cols].sum().sort_values(ascending=False)
```

## 11. 결과 다운로드

```python
!zip -r hate_type_results.zip \
  "분석 데이터/hate_predictions" \
  "분석 데이터/hate_type_predictions" \
  models/kc-electra-komultitext-binary/threshold.json \
  models/kc-electra-komultitext-binary/training_metrics.json \
  models/kc-electra-komultitext-hate-type/label_config.json \
  models/kc-electra-komultitext-hate-type/training_metrics.json
```

```python
from google.colab import files
files.download("hate_type_results.zip")
```
