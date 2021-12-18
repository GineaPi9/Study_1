import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import random

#SEED = 1234
#random.seed(SEED)

pd.set_option('mode.chained_assignment',  None)

# 데이터 불러오기
train = pd.read_excel('./train.xlsx')

# 데이터 재구성
train['rooms_per_bedrooms'] = train['total_rooms'] / train['total_bedrooms']
train['households_per_rooms'] = train['households'] / train['total_rooms']
train['households_per_population'] = train['households'] / train['population']

train = train.drop('total_rooms',axis=1)
train = train.drop('total_bedrooms',axis=1)
train = train.drop('households',axis=1)
train = train.drop('population',axis=1)
train = train.drop('housing_median_age', axis=1)

# target 분리
train = train.drop('Unnamed: 0',axis=1)
data = train.drop('target', axis=1)
target = train['target']

print(data.columns)

# val_set 분리
train_set, val_set, train_label, val_label = train_test_split(data, target, test_size = 0.2)

# 데이터 전처리
scaler = StandardScaler()
labelEncoder = LabelEncoder()
train_set["rooms_per_bedrooms"].fillna(0, inplace=True)
val_set["rooms_per_bedrooms"].fillna(0, inplace=True)
train_set['ocean_proximity'] = labelEncoder.fit_transform(train_set['ocean_proximity'])
val_set['ocean_proximity'] = labelEncoder.fit_transform(val_set['ocean_proximity'])
print(scaler.fit(train_set))
train_data_standardScaled = scaler.transform(train_set)
test_data_standardScaled = scaler.transform(val_set)

# Regression 모델 정의 및 사용
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#reg = LinearRegression()
#reg = Ridge()
#reg = BayesianRidge()
#reg = svm.SVR()
#reg = DecisionTreeRegressor()
reg = RandomForestRegressor(n_estimators=50)
reg.fit(train_data_standardScaled, train_label)

# 결과 확인
some_data = train_data_standardScaled[:5]
some_labels = val_label.iloc[:5]

print("예측: ", reg.predict(some_data))
print("정답: ", list(some_labels))

# train_set 성능
housing_predictions = reg.predict(train_data_standardScaled)
lin_mse = mean_squared_error(train_label, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("train_set: ", lin_rmse)

# val_set 성능
housing_predictions_test = reg.predict(test_data_standardScaled)
lin_mse = mean_squared_error(val_label, housing_predictions_test)
lin_rmse = np.sqrt(lin_mse)
print("val_set: ", lin_rmse)

# 모델 저장
from sklearn.externals import joblib
joblib.dump(reg, "my_model.pkl")

# 모델 load
my_model = joblib.load("my_model.pkl")

# load 된 모델로 train_set 성능
housing_predictions = my_model.predict(train_data_standardScaled)
lin_mse = mean_squared_error(train_label, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("train_set: ", lin_rmse)

# load 된 모델로 val_set 성능
housing_predictions_test = my_model.predict(test_data_standardScaled)
lin_mse = mean_squared_error(val_label, housing_predictions_test)
lin_rmse = np.sqrt(lin_mse)
print("val_set: ", lin_rmse)