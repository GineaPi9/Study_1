import numpy as np

def Per(A,i,j): # 행 바꿈 함수
    A[i],A[j] = A[j],A[i] # i+1행과 j+1행 바꿈
    return A


def Dia(A,i,c) : # i행에 scalar 값 c만큼 곱해주는 함수    
    col_length = len(A[i]) # 열의 개수 측정
    for j in range(col_length) : # i+1행 모든 열 원소에 상수 c를 곱함
        A[i][j] = c*(A[i][j]) 
    
    return A


def Ele(A,i,j,c) : # 행에 상수를 곱한 다른 행의 값을 더하거나 빼주는 함수
    
    row_length = len(A[i])    
    for k in range(row_length) : # i+1행k+1열 원소 - (j+1행 k+1열 원소)X(상수 c)
        A[i][k] = A[i][k] + c*(A[j][k])
    return A


def Gauss_Elimination(test_input):


    print("Matrix Input\n\n",np.matrix(test_input),"\n\n------------------------------------\n")

    row_len = len(test_input) # row의 개수
    col_len = len(test_input[0])  # column의 개수


    if(row_len <= col_len) : # 행의 길이가 열의 길이보다 작거나 같은 경우
        
        for i in range(0,row_len): # 행 개수 계산을 만큼 반복

            if(test_input[i][i] !=0 ):   # 입력 받은 행렬의 (i+1,i+1) 원소가 0이 아닌 경우
                Dia(test_input,i,(1/test_input[i][i])) # (i+1,i+1)원소 값으로 i행을 나눔 
                
            else :     # 입력 받은 행렬의 (i+1,i+1)원소가 0인 경우 
                        
                if(i+1<row_len) : # i+2행부터 0이 아닌 원소 찾기 위해 반복 
                    for k in range(i+1,row_len): # i+2행 부터 다시 확인
                        Per(test_input,i,k) # k+1행과 i+1행 변경
                        if(test_input[i][i]!=0): # 바뀐 행렬 (i+1,i+1)원소가 0인지 다시 확인 
                            Dia(test_input,i,(1/test_input[i][i])) # 0이 아닐 경우 행렬의 (i,i)원소 값으로 i행을 나눔
                            break #  행렬의 (i,i)원소값을 1로 만들어주었으니 탈출

                        else:
                            continue # i열의 모든 원소가 0인 경우 continue

        
            
            for j in range(0,i): # 1행부터 i행까지 반복
                Ele(test_input,j,i,-test_input[j][i]) # (i+1,i+1)원소와 같은 열, (i+1,i+1)원소 위의 원소들 모두를 0으로 만듦.

            for j in range(i+1,row_len): # i+2행부터 마지막 행까지 반복 
                Ele(test_input,j,i,-test_input[j][i]) # (i+1,i+1)원소와 같은 열, (i+1,i+1)원소 아래의 원소들 모두를 0으로 만듦.



    else : # 행 길이가 열 길이보다 큰 경우

        for i in range(0,col_len): # 열 개수 계산을 만큼 반복
                
            if(test_input[i][i] !=0 ):   # 입력 받은 행렬의 (i+1,i+1) 원소가 0이 아닌 경우
                Dia(test_input,i,(1/test_input[i][i])) # (i+1,i+1)원소 값으로 i+1행을 나눔 

            else :     # 입력 받은 행렬의 (i+1,i+1)원소가 0인 경우 
                        
                if(i+1<row_len) : # i+2행부터 0이 아닌 원소 찾기 위해 반복
                    for k in range(i+1,row_len): # i+2행부터 반복 해서 확인
                        Per(test_input,i,k) # k행과 i행 변경
                        
                        if(test_input[i][i]!=0): # 바뀐 행렬 (i+1,i+1)원소가 0인지 다시 확인 
                            Dia(test_input,i,(1/test_input[i][i])) # 0이 아닐 경우 행렬의 (i+1,i+1)원소 값으로 i행을 나눔                       
                            break #  행렬의 (i+1,i+1)원소값을 1로 만들어주었으니 탈출

                        else:
                            continue # i열의 모든 원소가 0인 경우 continue
            

            for j in range(0,i): # 1행부터 i행까지 반복
                Ele(test_input,j,i,-test_input[j][i]) # (i+1,i+1)원소와 같은 열, (i+1,i+1)원소 위의 원소들 모두를 0으로 만듦.

            for j in range(i+1,row_len): # i+2행부터 마지막 행까지 반복 
                Ele(test_input,j,i,-test_input[j][i]) # (i+1,i+1)원소와 같은 열, (i+1,i+1)원소 아래의 원소들 모두를 0으로 만듦.

    return np.matrix(test_input) # 행렬 형식으로 반환


input = [[0,0,0,4,5,6],[2,0,2,4,5,5],[2,4,1,8,5,3],[0,9,5,2,7,1]] # 입력 행렬 데이터

print("Result of Gauss Elimination\n(Reduced Echelon Form)\n\n",Gauss_Elimination(input)) # gauss elimination 결과 출력




# if(row_len <= col_len) : # 행의 개수가 열의 개수보다 크거나 같은 경우
    
#     for i in range(0,row_len): # 행 개수 계산을 만큼 반복


#         if(test_input[i][i] !=0 ):   # 입력 받은 행렬의 (i,i) 원소가 0이 아닌 경우
#             Dia(test_input,i,(1/test_input[i][i])) # (i,i)원소 값으로 i행을 나눔 
            
#         else :     # 입력 받은 행렬의 (i,i)원소가 0인 경우 
                    
#             if(i+1<row_len and test_input[i+1][i]!=0) : # 또한 (i+1,i)원소도 0이 아닌 경우 
#                 Per(test_input,i,i+1) # 바로 아래 행과 자리를 바꿔줌
#                 Dia(test_input,i,(1/test_input[i][i])) # 바뀐 행렬의 (i,i)원소 값으로 i행을 나눔

#             else : # 만약 바로 다음 행 (i+1,i)원소도 영인 경우
#                 for k in range(i+2,row_len): # i+2행 부터 다시 확인
#                     Per(test_input,i,k) # k행과 i행 변경
#                     if(test_input[i][i]!=0): # 바뀐 행렬 (i,i)원소가 0인지 다시 확인 
#                         Dia(test_input,i,(1/test_input[i][i])) # 0이 아닐 경우 행렬의 (i,i)원소 값으로 i행을 나눔
#                         print(np.matrix(test_input),"\n\n\n")
#                         break #  행렬의 (i,i)원소값을 1로 만들어주었으니 탈출

#                     else:
#                         continue #


    

#         print(np.matrix(test_input),"\n")
        
#         for j in range(0,i):
#             Ele(test_input,j,i,-test_input[j][i])
#             print(np.matrix(test_input),"\n")


#         for j in range(i+1,row_len):
#             Ele(test_input,j,i,-test_input[j][i]) 
#             print(np.matrix(test_input),"\n")





# else : # 행의 개수가 열의 개수보다 작은 경우

#     for i in range(0,col_len): # 열 개수 계산을 만큼 반복
            
#         if(test_input[i][i] !=0 ):   # 입력 받은 행렬의 (i,i) 원소가 0이 아닌 경우
#             Dia(test_input,i,(1/test_input[i][i])) # (i,i)원소 값으로 i행을 나눔 

#         else :     # 입력 받은 행렬의 (i,i)원소가 0인 경우 
                    
#             if(i+1<row_len and test_input[i+1][i]!=0) : # 또한 (i+1,i)원소도 0이 아닌 경우 
#                 Per(test_input,i,i+1) # 바로 아래 행과 자리를 바꿔줌
#                 Dia(test_input,i,(1/test_input[i][i])) # 바뀐 행렬의 (i,i)원소 값으로 i행을 나눔

#             else : # 만약 바로 다음 행 (i+1,i)원소도 0인 경우
#                 for k in range(i+2,row_len): # i+2행부터 반복 해서 확인
#                     Per(test_input,i,k) # k행과 i행 변경
#                     if(test_input[i][i]!=0): # 바뀐 행렬 (i,i)원소가 0인지 다시 확인 
#                         Dia(test_input,i,(1/test_input[i][i])) # 0이 아닐 경우 행렬의 (i,i)원소 값으로 i행을 나눔
#                         print(np.matrix(test_input),"\n\n\n")
#                         break #  행렬의 (i,i)원소값을 1로 만들어주었으니 탈출

#                     else:
#                         break # 


#         print(np.matrix(test_input),"\n")
        

#         for j in range(0,i):
#             Ele(test_input,j,i,-test_input[j][i])
#             print(np.matrix(test_input),"\n")


#         for j in range(i+1,row_len):
#             Ele(test_input,j,i,-test_input[j][i]) 
#             print(np.matrix(test_input),"\n")

     




