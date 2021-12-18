import numpy as np # 행렬/배열 형태를 사용하기 위한 라이브러리
from scipy import optimize  # opimize.minimize function을 사용하기 위한 라이브러리

# input
Q = np.array([[2., 1.], [1., 4.]])
c = np.array([5., 3.])
A = np.array([[1.,0.], [-1.,0.], [0.,1.], [0.,-1.]])
b = np.array([1.,1.,1.,1.])


# Objective function
def obj_func(x):
    return 0.5*np.dot(np.dot(x.T,Q),x) + np.dot(c, x)

# Objective function를 x에 대해 미분한 함수
def dev(x):
    return np.dot(Q,x)+c.T


# Constraint 조건 function
def con_func(x): 
    return -np.dot(A,x)+b.T


 # 랜덤한 2차원 initial point 생성
init_point = np.random.randn(2)


# Optimization의 constraints 생성(dictionary 형식)
con = {'type':'ineq', 'fun': con_func} # optimize.minimize 함수에 사용될 constraint condition을 dictionary 자료형으로 표현

# 최적화 결과
Opt_result = optimize.minimize(obj_func, init_point, jac=dev, method='SLSQP', constraints=con)
# optimize 함수를 이용하여 constraint condition이 주어진 objective function을 SLSQP 방법으로 최적화.
# SLSQP - Sequential Least Squares Programming 목적함수를 2차식으로 근사하여 반복적으로 문제를 풀어가는 알고리즘


# 최적과 결과 출력
# print(Opt_result)
print('Optimal solution x : {0}'.format([round(Opt_result['x'][0],5),round(Opt_result['x'][1],5)]))
print('Primal objective: {0}'.format(round(Opt_result['fun'],5)))


