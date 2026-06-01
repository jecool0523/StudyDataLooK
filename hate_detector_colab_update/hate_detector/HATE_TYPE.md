# 2-Stage Hate Type Classifier

`kc-electra-komultitext-binary` 모델로 먼저 유해 여부를 거른 뒤, 유해 문장이 어떤 유형의 혐오인지 분류하려면 별도의 멀티라벨 유형 모델을 학습합니다.

KoMultiText에서 `profanity`, `gender`, `politics`, `nation`, `race`, `region`, `generation`, `social_hierarchy`, `appearance`, `others` 중 하나라도 1인 문장만 남기고, 해당 10개 컬럼을 멀티라벨 정답으로 사용합니다.

## Train

```powershell
python -m hate_detector.train_hate_type --output-dir models/kc-electra-komultitext-hate-type --epochs 3
```

CPU에서 코드만 빠르게 확인하려면:

```powershell
python -m hate_detector.train_hate_type --output-dir models/smoke-kc-electra-hate-type --epochs 0.02 --max-train-samples 64 --max-eval-samples 64
```

## Predict

binary 예측이 끝난 CSV에 유형 예측을 붙이려면:

```powershell
python -m hate_detector.predict_hate_type_csv --type-model-dir models/kc-electra-komultitext-hate-type --input "분석 데이터/hate_predictions/skk_hate.csv" --output "분석 데이터/hate_type_predictions/skk_hate_typed.csv"
```

기본값은 CSV 안의 `is_harmful=1` 행에만 유형 모델을 적용합니다. 모든 행에 적용하려면 `--classify-all`을 추가합니다.

binary 예측 결과 폴더 전체에 적용하려면:

```powershell
python -m hate_detector.predict_hate_type_dir --type-model-dir models/kc-electra-komultitext-hate-type --input-dir "분석 데이터/hate_predictions" --output-dir "분석 데이터/hate_type_predictions"
```

추가되는 주요 컬럼:

- `primary_hate_type`: 가장 점수가 높은 혐오 유형
- `matched_hate_types`: 임계값 이상인 모든 혐오 유형
- `hate_type_<label>_score`: 유형별 확률 점수
- `hate_type_<label>`: 임계값 이상 여부
