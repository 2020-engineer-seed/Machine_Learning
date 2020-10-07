# ライブラリをロード
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# URLを作成
url = 'https://github.com/2020-engineer-seed/datasets/blob/master/Dataset_learn_1.csv'

# データセットをロード
dataframe = pd.read_csv(url)

# 標準化器を作成
starndardizer = StandardScaler()