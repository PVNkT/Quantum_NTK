import numpy as np

#normalize되어 나온 정답을 방정식에 대입하여 주어진 행렬 방정식의 해가 되기 위한 상수를 구하여 행렬 방정식의 해를 구한다. 
def get_answer(A, b, answer, hermition = False):
    #상수를 설정
    k = 1
    #주어진 행렬이 hermition인지 아닌지에 따라서 원래의 행렬 방정식을 복원하고 상수값을 구한다. 이때 각 성분에 따라서 k가 달라지기 때문에 평균을 취한다.
    if hermition:
        b_cal = np.matmul(A, answer)
        k_list = []
        #b 벡터의 요소중에 0이 포함되어 있을 경우 오류가 생길 수 있기 때문에 0인 부분을 제외한다.
        for i, b_el in enumerate(b):
            if b_el < 1e-7:
                pass
            else:
                k_list.append(b_el/b_cal[i])
        k = np.mean(np.array(k_list))
    else:
        A = A[:len(A[0])//2][:,len(A[0])//2:]
        b = b[:len(b)//2]
        answer = answer[len(answer)//2:]
        b_cal = np.matmul(A, answer)
        k_list = []
        #b 벡터의 요소중에 0이 포함되어 있을 경우 오류가 생길 수 있기 때문에 0인 부분을 제외한다.
        for i, b_el in enumerate(b):
            if b_el < 1e-7:
                pass
            else:
                k_list.append(b_el/b_cal[i])
        k = np.mean(np.array(k_list))
    return k*answer


if __name__ == '__main__':
    A = np.array([[1,2],[3,4]])
    b = np.array([5,6])
    A = np.vstack((np.hstack((np.zeros_like(A),A)),np.hstack((A.conj().T, np.zeros_like(A)))))
    b = np.hstack((b, np.zeros_like(b)))
    answer = np.array([-2.40224072e-15,  2.67374163e-15, -6.66509829e-01,  7.45496243e-01])
    print(get_answer(A, b, answer))