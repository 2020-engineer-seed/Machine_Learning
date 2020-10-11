# ライブラリをロード
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# URLを作成
url = 'https://github.com/2020-engineer-seed/datasets/blob/master/Dataset_learn_1.csv'

# データセットをロード
dataframe = pd.read_csv(url)

# 標準化器・推定器オブジェクトを作成
starndardizer = StandardScaler()

features_standardized = starndardizer.fit_transform(features)

#学習
rnn = RadiusNeighborsClassifier( radius = .5, n_job = -1 ).fit(features_standardized, target)

new_observations = [[1,1,1,1]]

#テストデータに対して予測

#評価