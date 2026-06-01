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
# 로컬 환경에서는 폰트를 설치하는 대신, 설치된 폰트를 사용하도록 설정합니다.
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
        # 리눅스에서는 'NanumGothic'을 찾습니다.
        # 만약 설치되어 있지 않다면, 사용자가 직접 설치해야 합니다.
        # (예: sudo apt-get install fonts-nanum)
        font_name = 'NanumGothic'
        
        # 폰트 매니저에서 폰트 검색
        if any(font.name == font_name for font in fm.fontManager.ttflist):
            plt.rc('font', family=font_name)
            print(f"Linux: '{font_name}' 폰트를 설정합니다.")
        else:
            print(f"Linux: '{font_name}' 폰트를 찾을 수 없습니다.")
            print("한글이 깨질 수 있습니다. '나눔고딕' 폰트를 설치해주세요. (예: sudo apt-get install fonts-nanum)")
            # 폰트가 없으면 일단 기본 sans-serif 사용
            plt.rc('font', family='sans-serif')

    # 폰트 설정 후 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)
    print("폰트 설정 완료.")

except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}")
    print("경고: 한글 폰트가 제대로 설정되지 않았을 수 있습니다. 그래프의 한글이 깨질 수 있습니다.")


# --- 2. 모델 및 토크나이저 로드 ---
print("모델 및 토크나이저를 로드합니다...")
model_name = 'beomi/kcbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define unsmile_labels and num_labels
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
# 로컬 GPU가 있으면 사용 (device=0), 없으면 CPU 사용 (device=-1)
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


# --- 4. 로컬 CSV 파일 로드 (수정된 부분) ---
# 'labeled_'로 시작하는 .csv 파일만 찾도록 수정
print("현재 디렉토리에서 'labeled_*.csv' 파일을 스캔합니다...")
target_pattern = 'labeled_*.csv'
filenames = glob.glob(target_pattern)

if not filenames:
    print(f"경고: '{target_pattern}' 패턴과 일치하는 CSV 파일을 찾을 수 없습니다. 스크립트와 같은 위치에 파일이 있는지 확인하세요.")
else:
    print(f"총 {len(filenames)}개의 CSV 파일 처리 대상: {filenames}")


# --- 5. 이전 결과 파일 정리 ---
prediction_column = '내용'

# Clean up any previously created predicted files and plot images
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
        # Load the CSV file into a pandas DataFrame
        try:
            csv_df = pd.read_csv(filename, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"  '{filename}' 파일 utf-8 디코딩 실패. cp949(EUC-KR)로 재시도합니다.")
            csv_df = pd.read_csv(filename, encoding='cp949')
        except FileNotFoundError:
            print(f"  오류: {filename} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        if prediction_column in csv_df.columns:
            predictions = []
            batch_size = 128  # 배치 크기 (메모리에 따라 조절)

            # 비어있거나(NaN) 문자열이 아닌 값을 미리 필터링
            texts_to_predict = [text for text in csv_df[prediction_column].tolist() if isinstance(text, str)]
            non_string_count = len(csv_df[prediction_column]) - len(texts_to_predict)
            
            if non_string_count > 0:
                print(f"  경고: '{prediction_column}' 열에서 문자열이 아닌 값 {non_string_count}개를 건너뜁니다.")

            if not texts_to_predict:
                print(f"  경고: '{prediction_column}' 열에 처리할 텍스트 데이터가 없습니다. 이 파일을 건너뜁니다.")
                continue

            print(f"  총 {len(texts_to_predict)}개의 텍스트에 대해 예측을 수행합니다...")
            # tqdm을 사용하여 진행 상황 표시
            for i in tqdm.tqdm(range(0, len(texts_to_predict), batch_size), desc=f"  {filename} 예측 중"):
                batch_texts = texts_to_predict[i:i + batch_size]
                
                # 파이프라인 실행
                try:
                    batch_predictions = pipe(batch_texts, truncation=True, padding='longest')
                except Exception as pipe_e:
                    print(f"    배치 처리 중 오류 발생: {pipe_e}. 이 배치를 건너뜁니다.")
                    # 오류 발생 시 빈 결과로 채움
                    batch_predictions = [[{'label': label, 'score': 0.0} for label in unsmile_labels]] * len(batch_texts)


                # 파이프라인 결과 형식 확인 및 저장
                if isinstance(batch_predictions, list) and len(batch_predictions) > 0:
                    if isinstance(batch_predictions[0], dict):
                        # (예: [{'label': '..', 'score': ..}, ...])
                        predictions.extend([[pred] for pred in batch_predictions])
                    elif isinstance(batch_predictions[0], list):
                         # (예: [[{'label': '..', 'score': ..}, ...], ...])
                        predictions.extend(batch_predictions)
                elif isinstance(batch_predictions, list) and len(batch_predictions) == 0:
                    pass # 빈 배치
                else:
                    print(f"  경고: 파이프라인 출력이 예상과 다릅니다. ({filename})")

            # 원본 DataFrame의 길이에 맞게 예측 결과 재구성 (NaN/비문자열 값 처리)
            full_predictions = []
            pred_idx = 0
            for text in csv_df[prediction_column].tolist():
                if isinstance(text, str) and pred_idx < len(predictions):
                    full_predictions.append(predictions[pred_idx])
                    pred_idx += 1
                else:
                    # 원본이 문자열이 아니거나 예측 결과가 부족한 경우 0점으로 채움
                    full_predictions.append([{'label': label, 'score': 0.0} for label in unsmile_labels])

            # 예측 결과를 DataFrame 열로 변환
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

            # 가장 높은 점수의 레이블로 '내용_분류' 열 생성
            score_column_prefix = f'{prediction_column}_score_'
            score_columns = [col for col in csv_df.columns if col.startswith(score_column_prefix)]
            
            if score_columns:
                csv_df[f'{prediction_column}_분류'] = csv_df[score_columns].idxmax(axis=1).str.replace(score_column_prefix, '')
            else:
                print(f"  경고: '{filename}' 파일에 대한 점수 열을 생성하지 못했습니다. '내용_분류' 열을 만들 수 없습니다.")
                csv_df[f'{prediction_column}_분류'] = None

            # 예측 결과가 포함된 새 CSV 파일 저장
            # 원본 파일 이름(labeled_...)을 기반으로 predicted_labeled_... 로 저장
            output_filename = f"predicted_{filename}" 
            csv_df.to_csv(output_filename, index=False, encoding='utf-8-sig') # 엑셀에서 한글이 깨지지 않도록 utf-8-sig 사용
            print(f"  처리 완료. 결과를 '{output_filename}' 파일로 저장했습니다.")

        else:
            print(f"  경고: '{prediction_column}' 열을 '{filename}' 파일에서 찾을 수 없습니다. 이 파일을 건너뜁니다.")

    except Exception as e:
        print(f"  *** '{filename}' 파일 처리 중 심각한 오류 발생: {e} ***")


# --- 7. 결과 집계 및 그래프 생성 ---
print("\n--- 모든 파일 처리 완료. 결과 집계 및 그래프 생성을 시작합니다. ---")

predicted_filenames = [f for f in os.listdir('.') if f.startswith('predicted_labeled_') and f.endswith('.csv')]
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

    # 그래프 생성
    print("\n--- 그래프 생성 중 ---")
    if not aggregated_results:
        print("그래프를 생성할 집계 결과가 없습니다.")
    else:
        for filename, counts in aggregated_results.items():
            if not counts:
                print(f"경고: '{filename}' 파일에 대한 집계 데이터가 없어 그래프를 생성할 수 없습니다.")
                continue
                
            labels = counts.keys()
            sizes = counts.values()

            try:
                fig1, ax1 = plt.subplots(figsize=(10, 8)) # 그래프 크기 조절
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')  # 원형 유지

                plt.title(f"레이블 분포: {filename}")
                
                # 그래프를 이미지 파일로 저장 (예: predicted_labeled_file1.csv -> label_distribution_predicted_labeled_file1.png)
                output_image_filename = f"label_distribution_{os.path.splitext(filename)[0]}.png"
                plt.savefig(output_image_filename, bbox_inches='tight') # 라벨이 잘리지 않도록
                print(f"  '{filename}'에 대한 파이 차트를 '{output_image_filename}'으로 저장했습니다.")
                plt.close(fig1)  # 메모리 해제를 위해 그래프 닫기
            
            except Exception as e:
                print(f"  오류: '{filename}' 파일의 파이 차트 생성 중 오류 발생: {e}")
                if 'fig1' in locals():
                    plt.close(fig1) # 오류 발생 시에도 그래프 닫기

print("\n모든 작업이 완료되었습니다.")