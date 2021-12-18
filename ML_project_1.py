import pandas as pd # data frame을 위한 라이브러리
from pandas import Series, DataFrame # # Data frame 생성 및 이용을 위한 라이브러리
import numpy as np # 배열 형태 사용을 위한 라이브러리
from sklearn import linear_model # Linear regression, Ridge regression 모델을 위한 라이브러리
from sklearn.model_selection import train_test_split # train data, test data 나누기 위한 라이브러리
from sklearn.preprocessing import StandardScaler # 데이터 표준화를 위한 라이브러리
from sklearn.metrics import mean_squared_error, mean_absolute_error #sklearn library 성능 평가 지표 사용
from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression model을 위한 라이브러리
import matplotlib.pyplot as plt # Plotting을 위한 라이브러리
import seaborn as sns # Statistical data visualization을 위한 라이브러리
import random # random 함수 사용을 위한 라이브러리
from math import sqrt # Root 사용을 위한 라이브러리
import joblib # Model 저장을 위한 라이브러리

# 데이터 불러오기
df=pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/BostonHousing2.csv") #데이터 파일 불러오기
df.info() # data 현재 상태 출력



# 위치(위도, 경도에 따른 주택 가격)
sns.set_style('darkgrid') # 그래프 배경 설정
plt.figure(figsize=(6,6))
sns.scatterplot(data=df, x='LON', y='LAT', hue='CMEDV', marker='o', alpha=0.5)
plt.title("Location(Latitude, Longitude) vs. CMEDV")

# Correlation heatmap 출력
correlation_measure = df.corr() # Data 행 간의 correlation 측정
plt.figure(figsize=(13,13)) # 그래프 사이즈 지정
sns.heatmap(data = correlation_measure, annot=True, fmt = '.2f', cmap='RdBu') # heatmap 그려주는 함수 (correlation)

# CMEDV에 대한 correlation이 높은 순서대로 나열
print('\n',correlation_measure['CMEDV'].sort_values(ascending = False))

# 일부 Data를 CMEDV와 Scatter Plot을 통해 비교
plt.figure(figsize=(5,5))
sns.scatterplot(data=df, x='RM', y='CMEDV', alpha=0.7) # RM과 CMEDV의 correlation을 보여줄 수있는 Scatter plot 출력
plt.title("RM vs. CMEDV")

plt.figure(figsize=(5,5))
sns.scatterplot(data=df, x='LSTAT', y='CMEDV', alpha=0.7) # RM과 CMEDV의 correlation을 보여줄 수있는 Scatter plot 출력
plt.title("LSTAT vs. CMEDV")

plt.figure(figsize=(5,5))
sns.scatterplot(data=df, x='DIS', y='CMEDV', alpha=0.7) # DIS과 CMEDV의 correlation을 보여줄 수있는 Scatter plot 출력
plt.title("DIS vs. CMEDV")

# 불러온 데이터 변형
df_1=df.drop(columns=['CHAS','LAT','DIS']) # Data 중 dummy variable인 CHAS와 correlation이 낮은 LON(경도), LAT(위도), DIS 제거



# Regression 을 위한 data Standardization(normalization) - 평균 0, 표준편차 1 
used_data_col =  ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'] # Modeling에 사용할 data columns(LON,LAT,DIS)
df_1[used_data_col]= StandardScaler().fit_transform(df_1[used_data_col]) # Data에서 modeling하는데 사용될 data column들을 standardize

#Train model/Test model 분리
Param = df_1[used_data_col] # Parameter로 사용될 data(주택 가격에 영향을 주는 요소들) 
Target = df_1['CMEDV'] # Target data (주택 가격)
Param_train, Param_test, Target_train, Target_test = train_test_split(Param,Target,test_size=0.2, random_state=3) # Data의 순서를 3번 섞은 후 Test, Train set을 1:4 비율로 나눔 

print("\nParameter train data 크기 : {0}\nTarget data 크기 : {1}\n".format(Param_train.shape, Target_train.shape)) # 나눈 Train set 크기 확인 
print("Train data 크기 : {0}\nTest data 크기 : {1}\n".format(Param_test.shape, Target_test.shape)) # 나눈 Test set 크기 확인




# 여러 모델을 이용한 예측 data 생성

# -------모델 1 : Linear Regression model------------------------------------------------
LinReg = linear_model.LinearRegression() # Model 1 : Lineaer Regression model 
model_1 = LinReg.fit(Param_train, Target_train) # Train data를 이용해 Linear Regression model을 훈련
predicted_TestSet_1 = LinReg.predict(Param_test) # 훈련된 Linear Regression model에 test data 입력 -> 예측 data 구함 

# 모델을 이용해 예측한 CMEDV와 실제 test data CMEDV 비교를 위한 Data 생성
LinReg_Compare_Data = pd.DataFrame({'Actual_Value':Target_test, 'Predicted_Value':predicted_TestSet_1}) # Data frame 1열 : 실제 data, 2열 : 예측 data
LinReg_Compare_Data = LinReg_Compare_Data.sort_values(by='Actual_Value').reset_index(drop=True) # 실제 data를 기준으로 오름차순으로 행을 정렬

# 예측 CMEDV와 실제 CMEDV(test data) 비교 그래프
plt.figure(figsize=(7, 7))
plt.scatter(LinReg_Compare_Data.index, LinReg_Compare_Data['Predicted_Value'], marker='x', color='crimson')
plt.scatter(LinReg_Compare_Data.index, LinReg_Compare_Data['Actual_Value'], color='black',  alpha=0.7)
plt.title("Prediction by Linear Regression & Actual Value Compare(Test set)") 
plt.legend(['Prediction_Value', 'Actual_Value']) # 범례


# -------모델 2 : Ridge Regression model------------------------------------------------
RidgeReg = linear_model.Ridge() # Model 2 : Ridge Regression model 
model_2 = RidgeReg.fit(Param_train, Target_train) # Train data를 이용해 Ridge Regression model을 훈련
predicted_TestSet_2 = RidgeReg.predict(Param_test) # 훈련된 Ridge Regression model에 test data 입력 -> 예측 data 구함 

# 모델을 이용해 예측한 CMEDV와 실제 test data CMEDV 비교를 위한 Data 생성
RidgeReg_Compare_Data = pd.DataFrame({'Actual_Value':Target_test, 'Predicted_Value':predicted_TestSet_2}) # Data frame 1열 : 실제 data, 2열 : 예측 data
RidgeReg_Compare_Data = RidgeReg_Compare_Data.sort_values(by='Actual_Value').reset_index(drop=True) # 실제 data를 기준으로 오름차순으로 행을 정렬

# 예측 CMEDV와 실제 CMEDV(test data) 비교 그래프
plt.figure(figsize=(7, 7))
plt.scatter(RidgeReg_Compare_Data.index, RidgeReg_Compare_Data['Predicted_Value'], marker='x', color='darkorange')
plt.scatter(RidgeReg_Compare_Data.index, RidgeReg_Compare_Data['Actual_Value'], color='black',  alpha=0.7)
plt.title("Prediction by Ridge Regression & Actual Value Compare(Test set)") 
plt.legend(['Prediction_Value', 'Actual_Value']) # 범례


# -------모델 3 : Decision Tree Regression model-------------------------------------------
DecTreeReg = DecisionTreeRegressor(max_depth=5) # Model 3 : Decision Tree Regression model 
model_3 = DecTreeReg.fit(Param_train, Target_train) # Train data를 이용해 Decision Tree Regression model을 훈련
predicted_TestSet_3 = DecTreeReg.predict(Param_test) # 훈련된 Decision Tree Regression model에 test data 입력 -> 예측 data 구함

# 모델을 이용해 예측한 CMEDV와 실제 test data CMEDV 비교를 위한 Data 생성
DecTreeReg_Compare_Data = pd.DataFrame({'Actual_Value':Target_test, 'Predicted_Value':predicted_TestSet_3}) # Data frame 1열 : 실제 data, 2열 : 예측 data
DecTreeReg_Compare_Data = DecTreeReg_Compare_Data.sort_values(by='Actual_Value').reset_index(drop=True) # 실제 data를 기준으로 오름차순으로 행을 정렬

# 예측 CMEDV와 실제 CMEDV(test data) 비교 그래프
plt.figure(figsize=(7, 7))
plt.scatter(DecTreeReg_Compare_Data.index, DecTreeReg_Compare_Data['Predicted_Value'], marker='x', color='blue')
plt.scatter(DecTreeReg_Compare_Data.index, DecTreeReg_Compare_Data['Actual_Value'], color='black',  alpha=0.7)
plt.title("Prediction by Decision Tree Regression & Actual Value Compare(Test set)") 
plt.legend(['Prediction_Value', 'Actual_Value']) # 범례


plt.figure(figsize=(7, 7))
plt.scatter(DecTreeReg_Compare_Data.index, DecTreeReg_Compare_Data['Actual_Value'], color='black',  alpha=0.2)
plt.scatter(LinReg_Compare_Data.index, LinReg_Compare_Data['Predicted_Value'], marker='x', color='crimson')
plt.scatter(RidgeReg_Compare_Data.index, RidgeReg_Compare_Data['Predicted_Value'], marker='x', color='darkorange')
plt.scatter(DecTreeReg_Compare_Data.index, DecTreeReg_Compare_Data['Predicted_Value'], marker='x', color='blue') 
plt.legend(['Actual_Value', 'Lin_Reg', 'Lin_Ridge_Reg', 'Dec_Tree_Reg']) # 범례



plt.show() # 그래프 출력 함수




# 성능 평가

# --------모델 1 : Linear Regression model 성능 평가------------------------------------------------
# 성능 평가 1 : MAE(Mean Absolute Error) [ 작을 수록(0에 가까울 수록) data에 fit ] 
predicted_TrainSet_1 = LinReg.predict(Param_train) # Train set 성능 평가를 위해 train set을 이용한 예측 data set 생성
print("\n---Linear Regression 성능 평가---\n")
print("MAE을 통한 Train set 성능평가 :\n{0}".format(mean_absolute_error(Target_train,predicted_TrainSet_1))) # Training set
print("MAE을 통한 Test set 성능평가 :\n{0}\n".format(mean_absolute_error(Target_test,predicted_TestSet_1))) # Test set

# 성능 평가 2 : RMSE(Root Mean Squared Error) [ 작을 수록(0에 가까울 수록) data에 fit ]
print("RMSE를 통한 Train set 성능평가 :\n{0}".format(sqrt(mean_squared_error(Target_train,predicted_TrainSet_1)))) # Training set
print("RMSE를 통한 Test set 성능평가 :\n{0}\n".format(sqrt(mean_squared_error(Target_test,predicted_TestSet_1)))) # Test set

# 성능 평가 3 : R-Squared [ 1에 가까울수록 data에 fit ]
print("R-Squared을 통한 Train set 성능평가 :\n{0}".format(model_1.score(Param_train,Target_train))) # Training set
print("R-Squared을 통한 Test set 성능평가 :\n{0}\n".format(model_1.score(Param_test,Target_test))) # Test set


# --------모델 2 : Ridge Regression model 성능 평가--------------------------------------------------
# 성능 평가 1 : MAE(Mean Absolute Error) [ 작을 수록(0에 가까울 수록) data에 fit ] 
predicted_TrainSet_2 = RidgeReg.predict(Param_train) # Train set 성능 평가를 위해 train set을 이용한 예측 data set 생성
print("---Ridge Regression 성능 평가---\n")
print("MAE을 통한 Train set 성능평가 :\n{0}".format(mean_absolute_error(Target_train,predicted_TrainSet_2))) 
print("MAE을 통한 Test set 성능평가 :\n{0}\n".format(mean_absolute_error(Target_test,predicted_TestSet_2))) 

# 성능 평가 2 : RMSE(Root Mean Squared Error) [ 작을 수록(0에 가까울 수록) data에 fit ]
print("RMSE를 통한 Train set 성능평가 :\n{0}".format(sqrt(mean_squared_error(Target_train,predicted_TrainSet_2)))) 
print("RMSE를 통한 Test set 성능평가 :\n{0}\n".format(sqrt(mean_squared_error(Target_test,predicted_TestSet_2)))) 

# 성능 평가 3 : R-Squared [ 1에 가까울수록 data에 fit ]
print("R-Squared을 통한 Train set 성능평가 :\n{0}".format(model_2.score(Param_train,Target_train))) 
print("R-Squared을 통한 Test set 성능평가 :\n{0}\n".format(model_2.score(Param_test,Target_test))) 


# --------모델 3 : Decision Tree Regression model 성능 평가--------------------------------------------
# 성능 평가 1 : MAE(Mean Absolute Error) [ 작을 수록(0에 가까울 수록) data에 fit ] 
predicted_TrainSet_3 = DecTreeReg.predict(Param_train) # Train set 성능 평가를 위해 train set을 이용한 예측 data set 생성
print("---Decision Tree Regression 성능 평가---\n")
print("MAE을 통한 Train set 성능평가 :\n{0}".format(mean_absolute_error(Target_train,predicted_TrainSet_3))) 
print("MAE을 통한 Test set 성능평가 :\n{0}\n".format(mean_absolute_error(Target_test,predicted_TestSet_3))) 

# 성능 평가 2 : RMSE(Root Mean Squared Error) [ 작을 수록(0에 가까울 수록) data에 fit ]
print("RMSE를 통한 Train set 성능평가 :\n{0}".format(sqrt(mean_squared_error(Target_train,predicted_TrainSet_3)))) 
print("RMSE를 통한 Test set 성능평가 :\n{0}\n".format(sqrt(mean_squared_error(Target_test,predicted_TestSet_3)))) 

# 성능 평가 3 : R-Squared [ 1에 가까울수록 data에 fit ]
print("R-Squared을 통한 Train set 성능평가 :\n{0}".format(model_3.score(Param_train,Target_train))) 
print("R-Squared을 통한 Test set 성능평가 :\n{0}\n".format(model_3.score(Param_test,Target_test))) 


# 모델 저장
joblib.dump(model_3,"DecTreeReg_Model_20171516.pkl")

