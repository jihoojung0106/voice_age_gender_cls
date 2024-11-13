import pandas as pd

# CSV 파일 읽기
csv_file = "mozilla_csv/train_filtered.csv"  # CSV 파일 경로
data = pd.read_csv(csv_file)

# gender와 age를 조합한 새로운 열 생성
data['gender_age'] = data['age'] + "_" + data['gender']

# 각 조합별 데이터 개수 세기
counts = data['gender_age'].value_counts()

# 결과 출력
print(counts)

# 결과를 DataFrame으로 저장하려면:
counts_df = counts.reset_index()
counts_df.columns = ['gender_age', 'count']
counts_df.to_csv("gender_age_counts.csv", index=False)
