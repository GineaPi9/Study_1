import numpy as np
from scipy import optimize

# input
Q = np.array([[1., 0.], [0., 0.]])
c = np.array([3., 4.])
A = np.array([[-1.,0.], [0.,-1.], [-1.,-3.], [2.,5.], [3.,4.]])
b = np.array([0.,0.,-15.,100.,80.])


# Objective function
def obj_func(x):
    return 0.5*np.dot(np.dot(x.T,Q),x) + np.dot(c, x)

# Objective function를 x에 대해 미분한 함수
def dev(x):
    return np.dot(Q,x)+c

def con_func(x):
    return -np.dot(A,x)+b

init_point = np.random.randn(2) # 랜덤한 initial point 생성

con = {'type':'ineq', 'fun': con_func} # constraints 생성
# cons = {'type':'ineq', 'fun':lambda x: b - np.dot(A,x), 'jac':lambda x: -A}
# 파이썬에서는 람다함수를 통해 이름이 없는 함수를 만들 수 있습니다.

Opt_result = optimize.minimize(obj_func, init_point, jac=dev, method='SLSQP', constraints=con)

# SLSQP - Sequential Least Squares Programming 목적함수를 2차식으로 근사하여 문제를 푸는 알고리즘
# a sequential least squares programming algorithm which uses the Han–Powell quasi–Newton method with a BFGS update of the B–matrix and an L1–test function in the step–length algorithm
# 'ineq'는 const 함수의 >= 일때를 고려


# print(obj_func)
print('Minimum x : {0}'.format(Opt_result['x']))
print('Objective function with minimum x : {0}'.format(Opt_result['fun']))
