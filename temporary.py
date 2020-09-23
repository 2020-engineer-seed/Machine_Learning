import pandas as pd
import numpy as np
from sklearn.modenl_selection import train_test_split

#データ読み込み
set_deta = pd.read_csv('DataSet_2020_09_18_16_46_43.csv',encoding="SHIFT-JIS")

#データをラベルと入力データに分離する

#学習用とテスト用に分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle =True)

#学習する

#評価する