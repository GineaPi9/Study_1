import numpy as np
import math 


print('첫 번째 행렬 a 입력(3차원) :')
a = np.array([list(map(float, input().split())) for i in range(3)]) # 임의의 벡터 a 입력
print(f'\n입력한 행렬 : \n{np.array(a)}\n')

print('두 번째 행렬 b 입력(3차원) :')
b = np.array([list(map(float, input().split())) for i in range(3)])  # 임의의 벡터 b 입력
print(f'\n입력한 행렬 : \n{np.array(b)}\n')


u1 = a # 첫번째 orthogonal basis 
u2 = b-(np.dot(b.T,u1)/np.dot(u1.T,u1))*u1 # 두번째 orthogonal basis 

norm_u1 = float(math.sqrt(np.dot(u1.T,u1))) # 첫번째 orthogonal basis의 norm 값
norm_u2 = float(math.sqrt(np.dot(u2.T,u2))) # 두번째 orthogonal basis의 norm 값

u1_onb = u1/norm_u1 # 첫번째 orthogonal 벡터를 norm으로 나누어 orthonomal basis 생성
u2_onb = u2/norm_u2 # 두번째 orthogonal 벡터를 norm으로 나누어 orthonomal basis 생성


print(f"첫번째 orthogonal 벡터 'u1' :\n {u1_onb}\n")
print(f"두번째 orthogonal 벡터 'u2' :\n {u2_onb}\n")
print("dot(u1,u2) = {0}".format(round(float(np.dot(u1.T,u2)),2))) # u1과 u2 벡터 내적 = 0 결과 확인 및 출력








# a = np.array([[3, 3, 5]]) # 임의의 벡터 a
# print(f"First input matrix 'a' :\n {a.T}\n")

# b = np.array([[3, 1, 9]]) # 임의의 벡터 b
# print(f"Second input matrix 'b' :\n {a.T}\n") 

# u1 = a.T # 첫번째 orthogonal basis 
# u2 = b.T-(np.dot(b,u1)/np.dot(u1.T,u1))*u1 # 두번째 orthogonal basis 
