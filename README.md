import os

# 保存先のディレクトリを指定します
save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'dark_horse_app.py')

code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# 仮想データセットの作成
data = {
    'RaceID': [1, 1, 1, 2, 2, 2],
    'HorseID': [101, 102, 103, 201, 202, 203],
    'HorseName': ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E', 'Horse F'],
    'Jockey': ['Jockey X', 'Jockey Y', 'Jockey Z', 'Jockey W', 'Jockey V', 'Jockey U'],
    'Trainer': ['Trainer M', 'Trainer N', 'Trainer O', 'Trainer P', 'Trainer Q', 'Trainer R'],
    'Odds': [5.0, 10.0, 15.0, 2.5, 8.0, 20.0],
    'FinishPosition': [1, 2, 3, 1, 2, 3]
}

df = pd.DataFrame(data)

# データのクレンジング
df = df.dropna()

# 特徴量エンジニアリング
df['Win'] = df['FinishPosition'].apply(lambda x: 1 if x == 1 else 0)

# 必要な特徴量の選択
features = ['Odds', 'Win']
df_features = df[features]

# データの分割
X = df_features[['Odds']]
y = df_features['Win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルのトレーニング
model = LogisticRegression()
model.fit(X_train, y_train)

# 勝利確率の計算
df['WinProbability'] = model.predict_proba(df[['Odds']])[:, 1]

# オッズが高く、勝利確率が一定以上の馬を穴馬として抽出
threshold_probability = 0.2
high_odds_threshold = 10.0

df['IsDarkHorse'] = df.apply(lambda row: row['WinProbability'] > threshold_probability and row['Odds'] > high_odds_threshold, axis=1)

dark_horses = df[df['IsDarkHorse']]

# Streamlitインターフェースの作成
st.title('穴馬抽出システム')
if st.button('穴馬を抽出'):
    st.write(dark_horses[['HorseName', 'Odds', 'WinProbability']])
"""

with open(save_path, 'w') as file:
    file.write(code)

print(f"Script saved to {save_path}")

