# Colab 실행 가이드

KoMultiText + KcELECTRA binary 유해표현 탐지기를 Colab GPU에서 학습하고, `crawler/dc/` 전체 CSV를 분석하는 순서입니다.

## 1. Colab 런타임 설정

Colab 상단 메뉴에서:

```text
런타임 > 런타임 유형 변경 > 하드웨어 가속기 > GPU
```

GPU 확인:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. 프로젝트 업로드

현재 프로젝트 폴더를 Google Drive에 올립니다. 예시는 아래 경로를 사용합니다.

```text
MyDrive/HEAR/code
```

Colab에서 Drive 마운트:

```python
from google.colab import drive
drive.mount("/content/drive")
```

프로젝트 폴더로 이동:

```python
%cd "/content/drive/MyDrive/HEAR/code"
```

## 3. 패키지 설치

```python
!pip install -q -r requirements-hate-detector.txt
```

설치 확인:

```python
import torch, transformers, datasets, sklearn
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("cuda", torch.cuda.is_available())
```

## 4. 학습 Smoke Test

전체 학습 전에 코드가 정상 동작하는지 작은 샘플로 확인합니다.

```python
!python -m hate_detector.train_binary \
  --output-dir models/smoke-kc-electra \
  --epochs 0.01 \
  --max-train-samples 16 \
  --max-eval-samples 16 \
  --batch-size 4 \
  --max-length 64
```

## 5. 전체 학습

GPU 기준 권장 시작값입니다.

```python
!python -m hate_detector.train_binary \
  --output-dir models/kc-electra-komultitext-binary \
  --epochs 3 \
  --batch-size 16 \
  --max-length 160 \
  --harmful-class-weight 1.8
```

GPU 메모리가 부족하면 `--batch-size 8`로 낮춥니다.

1 epoch만 먼저 돌린 뒤 총 3 epoch까지 이어서 학습하려면:

```python
!python -m hate_detector.train_binary \
  --output-dir models/kc-electra-komultitext-binary \
  --epochs 3 \
  --batch-size 16 \
  --max-length 160 \
  --harmful-class-weight 1.8 \
  --resume-from-checkpoint
```

## 6. Recall 우선 임계값 튜닝

검증셋에서 `recall >= 0.90`을 만족하는 후보 중 F1이 가장 높은 threshold를 저장합니다.

```python
!python -m hate_detector.tune_threshold \
  --model-dir models/kc-electra-komultitext-binary \
  --min-recall 0.90 \
  --batch-size 64 \
  --max-length 160
```

저장된 threshold 확인:

```python
import json
with open("models/kc-electra-komultitext-binary/threshold.json", encoding="utf-8") as f:
    print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
```

## 7. `crawler/dc/` 전체 분석

`all_schools.csv`가 개별 학교 CSV의 합본이면 중복 집계를 피하기 위해 제외합니다.

```python
!python -m hate_detector.predict_dir \
  --model-dir models/kc-electra-komultitext-binary \
  --input-dir crawler/dc \
  --output-dir "분석 데이터/hate_predictions" \
  --exclude-name all_schools.csv \
  --batch-size 64 \
  --max-length 192
```

결과 확인:

```python
import pandas as pd
summary = pd.read_csv("분석 데이터/hate_predictions/summary.csv")
summary
```

유해 점수가 높은 글 확인:

```python
from pathlib import Path
import pandas as pd

files = sorted(Path("분석 데이터/hate_predictions").glob("*_hate.csv"))
dfs = [pd.read_csv(path) for path in files]
all_pred = pd.concat(dfs, ignore_index=True)

cols = ["source_file", "제목", "내용", "harmful_score", "model_score", "keyword_score", "matched_terms", "링크"]
all_pred.sort_values("harmful_score", ascending=False)[cols].head(30)
```

## 8. 결과 압축 다운로드

```python
!zip -r hate_predictions.zip "분석 데이터/hate_predictions" models/kc-electra-komultitext-binary/threshold.json models/kc-electra-komultitext-binary/training_metrics.json
```

```python
from google.colab import files
files.download("hate_predictions.zip")
```

## 추천 튜닝값

- 더 많이 잡기: `tune_threshold --min-recall 0.95`
- 오탐 줄이기: `tune_threshold --min-recall 0.85`
- GPU 메모리 부족: 학습 `--batch-size 8`, 추론 `--batch-size 32`
- 긴 글 반영 강화: 추론 `--max-length 256`
