import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
import torch
import matplotlib.font_manager as fm  # 폰트 관리를 위해 import
import platform  # 운영체제 확인을 위해 import
import glob  # 로컬 파일 검색을 위해 import

# --- 1. 운영체제별 한글 폰트 설정 ---
print("운영체제에 맞는 한글 폰트를 설정합니다...")
try:
    system_name = platform.system()
    if system_name == 'Windows':
        font_name = 'Malgun Gothic'
        plt.rc('font', family=font_name)
        print(f"Windows OS: '{font_name}' 폰트를 설정합니다.")
    elif system_name == 'Darwin':  # macOS
        font_name = 'AppleGothic'
        plt.rc('font', family=font_name)
        print(f"macOS: '{font_name}' 폰트를 설정합니다.")
    elif system_name == 'Linux':
        font_name = 'NanumGothic'
        if any(font.name == font_name for font in fm.fontManager.ttflist):
            plt.rc('font', family=font_name)
            print(f"Linux: '{font_name}' 폰트를 설정합니다.")
        else:
            print(f"Linux: '{font_name}' 폰트를 찾을 수 없습니다.")
            print("한글이 깨질 수 있습니다. '나눔고딕' 폰트를 설치해주세요. (예: sudo apt-get install fonts-nanum)")
            plt.rc('font', family='sans-serif')
    plt.rc('axes', unicode_minus=False)
    print("폰트 설정 완료.")
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}")
    print("경고: 한글 폰트가 제대로 설정되지 않았을 수 있습니다.")


# --- 2. 모델 및 토크나이저 로드 ---
print("모델 및 토크나이저를 로드합니다...")
model_name = 'beomi/kcbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

unsmile_labels = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오", "악플/욕설", "clean", "장애"]
num_labels = len(unsmile_labels)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.config.id2label = {i: label for i, label in enumerate(unsmile_labels)}
model.config.label2id = {label: i for i, label in enumerate(unsmile_labels)}


# --- 3. 파이프라인 초기화 ---
print("TextClassificationPipeline을 초기화합니다...")
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True,
    function_to_apply='sigmoid'
)
if torch.cuda.is_available():
    print("CUDA (GPU)를 사용하여 파이프라인 실행합니다.")
else:
    print("CUDA (GPU)를 찾을 수 없습니다. CPU로 파이프라인 실행합니다.")


# --- 4. 로컬 CSV 파일 로드 ---
print("현재 디렉토리에서 'labeled_*' 패턴의 파일을 스캔합니다...")
target_pattern = 'labeled_*'
all_found = glob.glob(target_pattern)

filenames = [f for f in all_found if not (
    f.startswith('predicted_') or 
    f.startswith('label_distribution_') or 
    f.endswith('.png') or 
    f.endswith('.py')
)]

if not filenames:
    print(f"경고: '{target_pattern}' 패턴과 일치하는 CSV 파일을 찾을 수 없습니다.")
    print("스크립트와 같은 위치에 'labeled_...'로 시작하는 파일이 있는지 확인하세요.")
else:
    print(f"총 {len(filenames)}개의 CSV 파일 처리 대상: {filenames}")


# --- 5. 이전 결과 파일 정리 ---
prediction_column = '내용'

for f in os.listdir('.'):
    if f.startswith('predicted_') and f.endswith('.csv'):
        try:
            os.remove(f)
            print(f"이전 예측 파일 삭제: {f}")
        except PermissionError:
            print(f"경고: {f} 파일을 삭제할 수 없습니다. (파일이 열려있을 수 있습니다)")
    if f.startswith('label_distribution_') and f.endswith('.png'):
        try:
            os.remove(f)
            print(f"이전 그래프 이미지 삭제: {f}")
        except PermissionError:
             print(f"경고: {f} 파일을 삭제할 수 없습니다. (파일이 열려있을 수 있습니다)")


# --- 6. 예측 수행 및 파일 저장 ---
for filename in filenames:
    print(f"\n파일 처리 시작: {filename}")
    try:
        try:
            csv_df = pd.read_csv(filename, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"  '{filename}' 파일 utf-8 디코딩 실패. cp949(EUC-KR)로 재시도합니다.")
            csv_df = pd.read_csv(filename, encoding='cp949')
        except FileNotFoundError:
            print(f"  오류: {filename} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
        except pd.errors.ParserError:
            print(f"  오류: {filename} 파일이 CSV 형식이 아닌 것 같습니다. 건너뜁니다.")
            continue

        if prediction_column in csv_df.columns:
            predictions = []
            batch_size = 128

            texts_to_predict = [text for text in csv_df[prediction_column].tolist() if isinstance(text, str)]
            non_string_count = len(csv_df[prediction_column]) - len(texts_to_predict)
            
            if non_string_count > 0:
                print(f"  경고: '{prediction_column}' 열에서 문자열이 아닌 값 {non_string_count}개를 건너뜁니다.")
            if not texts_to_predict:
                print(f"  경고: '{prediction_column}' 열에 처리할 텍스트 데이터가 없습니다. 이 파일을 건너뜁니다.")
                continue

            print(f"  총 {len(texts_to_predict)}개의 텍스트에 대해 예측을 수행합니다...")
            for i in tqdm.tqdm(range(0, len(texts_to_predict), batch_size), desc=f"  {filename} 예측 중"):
                batch_texts = texts_to_predict[i:i + batch_size]
                try:
                    batch_predictions = pipe(batch_texts, truncation=True, padding='longest')
                except Exception as pipe_e:
                    print(f"    배치 처리 중 오류 발생: {pipe_e}. 이 배치를 건너뜁니다.")
                    batch_predictions = [[{'label': label, 'score': 0.0} for label in unsmile_labels]] * len(batch_texts)

                if isinstance(batch_predictions, list) and len(batch_predictions) > 0:
                    if isinstance(batch_predictions[0], dict):
                        predictions.extend([[pred] for pred in batch_predictions])
                    elif isinstance(batch_predictions[0], list):
                        predictions.extend(batch_predictions)
                elif isinstance(batch_predictions, list) and len(batch_predictions) == 0:
                    pass
                else:
                    print(f"  경고: 파이프라인 출력이 예상과 다릅니다. ({filename})")

            full_predictions = []
            pred_idx = 0
            for text in csv_df[prediction_column].tolist():
                if isinstance(text, str) and pred_idx < len(predictions):
                    full_predictions.append(predictions[pred_idx])
                    pred_idx += 1
                else:
                    full_predictions.append([{'label': label, 'score': 0.0} for label in unsmile_labels])

            processed_predictions = []
            for prediction_list in full_predictions:
                scores_dict = {}
                if isinstance(prediction_list, list):
                    for item in prediction_list:
                        if isinstance(item, dict) and 'label' in item and 'score' in item:
                            scores_dict[f"{prediction_column}_score_{item['label']}"] = item['score']
                processed_predictions.append(scores_dict)

            prediction_df = pd.DataFrame(processed_predictions)
            csv_df = pd.concat([csv_df, prediction_df], axis=1)

            score_column_prefix = f'{prediction_column}_score_'
            score_columns = [col for col in csv_df.columns if col.startswith(score_column_prefix)]
            
            if score_columns:
                csv_df[f'{prediction_column}_분류'] = csv_df[score_columns].idxmax(axis=1).str.replace(score_column_prefix, '')
            else:
                print(f"  경고: '{filename}' 파일에 대한 점수 열을 생성하지 못했습니다.")
                csv_df[f'{prediction_column}_분류'] = None
                
            base_filename = os.path.basename(filename)
            base_name_no_ext = os.path.splitext(base_filename)[0]
            output_filename = f"predicted_{base_name_no_ext}.csv"
            
            csv_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"  처리 완료. 결과를 '{output_filename}' 파일로 저장했습니다.")

        else:
            print(f"  경고: '{prediction_column}' 열을 '{filename}' 파일에서 찾을 수 없습니다. 이 파일을 건너뜁니다.")

    except Exception as e:
        print(f"  *** '{filename}' 파일 처리 중 심각한 오류 발생: {e} ***")


# --- 7. 결과 집계 및 그래프 생성 (수정된 부분) ---
print("\n--- 모든 파일 처리 완료. 결과 집계 및 그래프 생성을 시작합니다. ---")

predicted_filenames = [f for f in os.listdir('.') if f.startswith('predicted_') and f.endswith('.csv')]
print(f"집계 및 그래프 생성을 위해 {len(predicted_filenames)}개의 예측 파일을 찾았습니다: {predicted_filenames}")

aggregated_results = {}

if not predicted_filenames:
    print("집계할 예측 파일이 없습니다.")
else:
    for filename in predicted_filenames:
        print(f"\n파일 집계 중: {filename}")
        try:
            csv_df = pd.read_csv(filename)

            if f'{prediction_column}_분류' in csv_df.columns:
                label_counts = csv_df[f'{prediction_column}_분류'].value_counts()
                aggregated_results[filename] = label_counts.to_dict()
                print(f"  '{filename}' 파일 레이블 집계 성공:")
                for label, count in label_counts.items():
                    print(f"    {label}: {count}")
            else:
                print(f"  오류: '{f'{prediction_column}_분류'}' 열이 {filename} 파일에 없습니다. 집계를 건너뜁니다.")

        except Exception as e:
            print(f"  오류: {filename} 파일 집계 중 오류 발생: {e}")

    # --- 그래프 생성 (수정된 부분) ---
    print("\n--- 그래프 생성 중 ('clean' 레이블 제외) ---")
    if not aggregated_results:
        print("그래프를 생성할 집계 결과가 없습니다.")
    else:
        for filename, counts in aggregated_results.items():
            
            # "clean" 레이블을 제외하기 위해 딕셔너리 복사 후 'clean' 키 삭제
            counts_without_clean = counts.copy()
            if 'clean' in counts_without_clean:
                del counts_without_clean['clean']
                print(f"  '{filename}' 그래프 생성: 'clean' 레이블 제외.")

            # 'clean' 제외 후 데이터가 없는 경우 건너뛰기
            if not counts_without_clean:
                print(f"  경고: '{filename}' 파일에서 'clean'을 제외한 집계 데이터가 없어 그래프를 생성할 수 없습니다.")
                continue
                
            labels = counts_without_clean.keys()
            sizes = counts_without_clean.values()

            try:
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                
                # 그래프 제목에 '(clean 제외)' 추가
                plt.title(f"레이블 분포: {filename} (clean 제외)")
                
                output_image_filename = f"label_distribution_{os.path.splitext(filename)[0]}_no_clean.png" # 파일명에도 표시
                plt.savefig(output_image_filename, bbox_inches='tight')
                print(f"  '{filename}'(clean 제외) 파이 차트를 '{output_image_filename}'으로 저장했습니다.")
                plt.close(fig1)
            
            except Exception as e:
                print(f"  오류: '{filename}' 파일의 파이 차트 생성 중 오류 발생: {e}")
                if 'fig1' in locals():
                    plt.close(fig1)

print("\n모든 작업이 완료되었습니다.")