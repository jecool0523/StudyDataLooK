#패키지 cmd에
#!pip install transformers
#pip install torch
#pip install pandas
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install wordcloud seaborn matplotlib
#pip install swifter

import pandas as pd
import chardet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import swifter
from transformers import BertForSequenceClassification
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

#한글 다 깨짐
import matplotlib.pyplot as plt
from matplotlib import rc

# 윈도우 기준 한글 폰트 설정
rc('font', family='Malgun Gothic')

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 모델 준비
model_name = 'smilegate-ai/kor_unsmile'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = TextClassificationPipeline(
     model=model,
     tokenizer=tokenizer,
     device=0,  # GPU 사용
     top_k=None  # return_all_scores 대체
)

#인코딩 ㅈ됨-방법좀 제발
import sys
sys.stdout.reconfigure(encoding='utf-8-sig')
#(encoding='fuck')

import os
os.system("chcp 65001")

#인코딩 sig넣으셈
df = pd.read_csv(r"C:\Users\seocheon\Desktop\dimigo\AI_EBS\dimigo_crawler\di.csv", encoding='utf-8-sig')

df["추천수"] = pd.to_numeric(df["추천수"].astype(str).str.replace("추천", "").str.strip(), errors="coerce")
df["조회수"] = pd.to_numeric(df["조회수"].astype(str).str.replace("조회", "").str.strip(), errors="coerce")

# 혐오 점수 추출 
def get_hate_scores(text):
    try:
        result = pipe(text[:512])[0]
        return {entry['label']: entry['score'] for entry in result}
    except:
        return {}

def detect_hate_bool(text):
    try:
        result = pipe(text[:512])[0]
        sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
        return 1 if sorted_result[0]['label'].lower() != 'clear' else 0
    except:
        return 0

# 제목
scores = df["제목"].astype(str).swifter.apply(get_hate_scores)
scores_df = pd.json_normalize(scores)
scores_df.columns = ["제목_" + col for col in scores_df.columns]
df = pd.concat([df, scores_df], axis=1)

# 내용
scores = df["내용"].astype(str).swifter.apply(get_hate_scores)
scores_df = pd.json_normalize(scores)
scores_df.columns = ["내용_" + col for col in scores_df.columns]
df = pd.concat([df, scores_df], axis=1)

# 혐오 표현 여부 판별 함수
def detect_hate_bool(text):
    try:
        result = pipe(text[:512])[0]
        sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
        return 1 if sorted_result[0]['label'].lower() != 'clear' else 0
    except:
        return 0

# 제목 기준 혐오여부 컬럼 생성
df["제목_혐오여부"] = df["제목"].astype(str).swifter.apply(detect_hate_bool)

# 내용 기준 혐오여부 컬럼 생성
df["내용_혐오여부"] = df["내용"].astype(str).swifter.apply(detect_hate_bool)


print("데이터 분석 및 시각화 시작...")

# 글쓴이
def simplify_author(author):
    if '디갤러' in str(author):
        return '디갤러'
    elif 'ㅇㅇ' in str(author):                 
        return 'ㅇㅇ'
    else:
        return '그 외'

df["글쓴이_단순"] = df["글쓴이"].apply(simplify_author)

# 상관관계
numeric_cols = [col for col in df.columns if (
    col.startswith("제목_") or col.startswith("내용_") or col in ["추천수", "조회수"]
)]
corr_df = df[numeric_cols]
corr = corr_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("상관관계 히트맵")
plt.savefig("상관관계_히트맵.png", dpi=300, bbox_inches='tight')
plt.show()


# 1. 워드클라우드
hate_df = df[df["내용_혐오여부"] == 1]
text = " ".join(hate_df["내용"].astype(str))
wordcloud = WordCloud(
    font_path=r"C:\Users\seocheon\Desktop\dimigo\AI_EBS\dimigo_crawler\GothicA1-Medium.ttf",  # 한글 = 지정 필수
    background_color="white",
    width=800,
    height=400
).generate(text)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("혐오 높은 글의 주요 키워드 워드클라우드")
plt.savefig("워드클라우드_혐오글.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. 산점도
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="내용_혐오여부",  #변경
    y="조회수",
    hue="글쓴이_단순",
    palette="Set1"
)
plt.title("혐오 점수 vs. 조회수 (글쓴이별)")
plt.savefig("산점도_혐오점수_vs_조회수.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. KDE Plot
plt.figure(figsize=(10, 6))
for group in df["글쓴이_단순"].unique():
    sns.kdeplot(
        data=df[df["글쓴이_단순"] == group],
        x="내용_혐오여부",
        label=group
    )
plt.title("혐오 점수 분포 (글쓴이별)")
plt.legend()
plt.savefig("KDE_혐오점수분포.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Pairplot
sns.pairplot(
    df,
    vars=["추천수", "조회수", "내용_혐오여부"],
    hue="글쓴이_단순",
    palette="Set1"
)
plt.suptitle("추천수, 조회수, 혐오 점수 관계 종합 시각화", y=1.02)
plt.savefig("Pairplot_종합관계.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. 시간 
df["날짜"] = pd.to_datetime(df["날짜"], errors='coerce')
daily_hate_ratio = df.groupby(df["날짜"].dt.date)["내용_혐오여부"].mean()
rolling_mean = daily_hate_ratio.rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_hate_ratio.index, daily_hate_ratio.values, marker="o", label="일별 혐오 비율")
plt.plot(daily_hate_ratio.index, rolling_mean, color="red", linewidth=2, label="7일 이동평균")
plt.title("날짜별 혐오 표현 비율 변화 (이동평균 포함)")
plt.ylabel("혐오 표현 비율")
plt.xlabel("날짜")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.savefig("시간흐름_혐오표현비율.png", dpi=300, bbox_inches='tight')
plt.show()

# 6
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="글쓴이_단순", y="내용_혐오여부", palette="Set1")
sns.swarmplot(data=df, x="글쓴이_단순", y="내용_혐오여부", color=".25")
plt.title("글쓴이별 내용_혐오여부 점수 분포")
plt.savefig("글쓴이별_내용_혐오여부_점수분포.png", dpi=300, bbox_inches='tight')
plt.show()

#저장
df.to_csv("AI!.csv", index=False, encoding="utf-8-sig")
print("전체 데이터가 포함된 CLS 파일 저장 완료.")