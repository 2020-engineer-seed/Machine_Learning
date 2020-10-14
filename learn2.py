# ライブラリのインポート
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# データのロード
data1 = pd.read_csv('Dataset_learn_2.csv')  # learn_1
data2 = pd.read_csv('Dataset_test_1.csv')  # test_1
data3 = pd.read_csv('Dataset_test_2.csv')  # test_2

# データの整理
# ------------------------------------------------------------------------------
tr = data1.pivot_table(values='RSSI',
                       index=data1[['SerialNumber', 'Latitude', 'Longitude']],
                       columns='APName')
m1 = tr.mean().astype('int64')
tr = tr.fillna(m1)
tr['Latitude'] = tr.index.get_level_values('Latitude')
tr['Longitude'] = tr.index.get_level_values('Longitude')
# ------------------------------------------------------------------------------
te1 = data2.pivot_table(values='RSSI',
                        index=data2[['SerialNumber', 'Latitude', 'Longitude']],
                        columns='APName')
m2 = te1.mean().astype('int64')
te1 = te1.fillna(m2)
te1['Latitude'] = te1.index.get_level_values('Latitude')
te1['Longitude'] = te1.index.get_level_values('Longitude')
# ------------------------------------------------------------------------------
te2 = data3.pivot_table(values='RSSI',
                        index=data3[['SerialNumber', 'Latitude', 'Longitude']],
                        columns='APName')
m3 = te2.mean().astype('int64')
te2 = te2.fillna(m3)
te2['Latitude'] = te2.index.get_level_values('Latitude')
te2['Longitude'] = te2.index.get_level_values('Longitude')
# ------------------------------------------------------------------------------
print(tr)
# trainデータとtestデータの作成
X_train = tr[['AP001', 'AP002', 'AP003', 'AP004', 'AP005']].values
X_test1 = te1[['AP001', 'AP002', 'AP003', 'AP004', 'AP005']].values
X_test2 = te2[['AP001', 'AP002', 'AP003', 'AP004', 'AP005']].values
Y_train = tr[['Latitude', 'Longitude']].values
Y_test1 = te1[['Latitude', 'Longitude']].values
Y_test2 = te2[['Latitude', 'Longitude']].values

# 説明変数の標準化
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test1 = sc.transform(X_test1)
X_test2 = sc.transform(X_test2)

# 勾配ブースティング
gbr = MultiOutputRegressor(GradientBoostingRegressor(subsample=0.5,
                                                     min_samples_split=4,
                                                     n_estimators=80))
gbr.fit(X_train, Y_train)

# 精度の検証(交差検証)
# 決定係数・平均平方二乗誤差・平均絶対誤差(test1)
Y_pre1 = gbr.predict(X_test1)
r2_1 = r2_score(Y_test1, Y_pre1)
rmse1 = np.sqrt(mean_squared_error(Y_test1, Y_pre1))
mae1 = mean_absolute_error(Y_test1, Y_pre1)
print('---test1----------------------')
print('r2', r2_1)
print('rmse', rmse1)
print('mae', mae1)
print('------------------------------' + '\n')

# 決定係数・平均平方二乗誤差・平均絶対誤差(test2)
Y_pre2 = gbr.predict(X_test2)
r2_2 = r2_score(Y_test2, Y_pre2)
rmse2 = np.sqrt(mean_squared_error(Y_test2, Y_pre2))
mae2 = mean_absolute_error(Y_test2, Y_pre2)
print('---test2----------------------')
print('r2', r2_2)
print('rmse', rmse2)
print('mae', mae1)
print('------------------------------' + '\n')

# learn1-test1の図の描写
plt.figure(figsize=(12, 8))
plt.scatter(Y_test1[:, 1], Y_test1[:, 0],
            s=7, alpha=0.2, color='red', label='test')  # 正解
plt.scatter(Y_pre1[:, 1], Y_pre1[:, 0],
            s=7, alpha=0.2, color='blue', label='predict')  # 予測
plt.title('learn2_test1_map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc="upper left", borderaxespad=0, fontsize=15)
plt.show()

# learn1-test2の図の描写
plt.figure(figsize=(12, 8))
plt.scatter(Y_test2[:, 1], Y_test2[:, 0],
            s=7, alpha=0.2, color='red', label='test')  # 正解
plt.scatter(Y_pre2[:, 1], Y_pre2[:, 0],
            s=7, alpha=0.2, color='blue', label='predict')  # 予測
plt.title('learn2_test2_map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc="upper left", borderaxespad=0, fontsize=15)
plt.show()
