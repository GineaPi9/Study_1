import numpy # 행렬/배열 형태를 사용하기 위한 라이브러리
from cvxopt import matrix # cvxopt : convex optimization을 위한 python package
from cvxopt import solvers

# Quadratic program parameter 행렬
Q = matrix(numpy.array([[2,1],[1,4]]), tc='d') # tc='d' : double 형태의 상수 matrix를 생성
c = matrix(numpy.array([5,3]), tc='d')
A = matrix(numpy.array([[1,0],[-1,0],[0,1],[0,-1]]), tc='d')
b = matrix(numpy.array([1,1,1,1]), tc='d') 

sol = solvers.qp(Q,c,A,b)

sol['x']
print(sol['x'])
print('Primal objective : {0}'.format(round(sol['primal objective'],4)))