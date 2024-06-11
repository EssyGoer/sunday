インポートOS

#保存先のディレクトリ
save_path = os.path.join(os.path.expanduser('〜')、'ドキュメント'、'dark_horse_app.py')

コード = """
pandasをpdとしてインポートする
sklearn.model_selection から train_test_split をインポートします
sklearn.linear_model から LogisticRegression をインポートします
streamlit を st としてインポートする

#テキストの作成
データ = {
    'レースID':[1、1、1、2、2、2]、
    '馬ID':[101、102、103、201、202、203]、
    '馬名':[「馬A」、「馬B」、「馬C」、「馬D」、「馬E」、「馬F」]、
    '騎手'：[「騎手 X」、「騎手 Y」、「騎手 Z」、「騎手 W」、「騎手 V」、「騎手 U」]、
    'トレーナー'：[「トレーナー M」、「トレーナー N」、「トレーナー O」、「トレーナー P」、「トレーナー Q」、「トレーナー R」]、
    「オッズ」:[5.0、10.0、15.0、2.5、8.0、20.0]、
    '終了位置':[1、2、3、1、2、3]
}

df = pd.DataFrame(データ)

#データのクレンジング
df = df.dropna()

#特徴量エンジニアリング
df['勝つ']=自由度['終了位置'].apply(lambda x: x == 1 の場合は 1、それ以外の場合は 0)

#必要な特徴量の選択
特徴 =[「オッズ」、「勝利」]
df_features = df[特徴]

#データの分割
X = df_features[[「オッズ」]]
y = 特徴量['勝つ']
X_train、X_test、y_train、y_test = train_test_split(X、y、test_size=0.2、random_state=42) です。

#モデルのトレーニング
モデル = ロジスティック回帰()
モデルをフィット(X_train, y_train)

#勝利確率の計算
df[「勝利確率」]= モデル.予測確率(df[[「オッズ」]]）[:, 1]

#オッズが高く、勝利確率が一定以上の馬を穴馬として抽出
閾値確率 = 0.2
高オッズ閾値 = 10.0

df[「ダークホース」]= df.apply(lambda 行: 行[「勝利確率」]> 閾値確率と行[「オッズ」]> 高オッズ閾値、軸=1)

ダークホース = df[df[「ダークホース」]]

#Streamlitインターフェースの作成
st.title('穴馬抽出システム')
if st.button('穴馬を抽出'):
    st.write(ダークホース[[「馬名」、「オッズ」、「勝率」]]）
「」

open(save_path, 'w') をファイルとして実行します:
    file.write(コード)

print(f"スクリプトが{save_path}に保存されました")
