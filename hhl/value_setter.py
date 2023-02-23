import numpy as np

#정확한 계산을 위해서 행렬 A의 eigenvalue를 고전적으로 계산하여 eigenvalue의 최대값, 최소값, 음의 값 존재 여부를 확인한다.
def get_eigenvalue(A):
    lambda_max = max(np.abs(np.linalg.eigvals(A)))
    lambda_min = min(np.abs(np.linalg.eigvals(A)))
    if np.any(np.linalg.eigvals(A)< 0):
        neg_vals = True
    else:
        neg_vals = False
    return lambda_max, lambda_min, neg_vals

#delta 값을 계산하기 위한 함수 
def get_delta(n_l, lambda_min, lambda_max, neg_vals):
    """
    주어진 고유값을 특정한 수(nl)의 bit의 이진법으로 표현하기 위해서 사용할 format string을 만듬. 
    qubit 하나는 음의 값을 표현하기 위해서 사용하기 때문에 빼줌
    2를 더하는 이유는 format을 사용하는 경우 앞에 0b라는 문자열이 추가되기 때문
    """ 
    formatstr = "#0" + str(n_l-neg_vals + 2) + "b"
    """
    eigenvalue의 최소값을 eigenvalue의 최대값으로 나누면 0~1 사이의 값을 가진다.
    아래의 코드는 이 0~1 사이의 값을 주어진 만큼의 bit를 사용하는 2진법 형태의 소수 표현법으로 표시하기 위한 코드이다. 
    """
    
    #우선 eigenvalue의 최소값을 최대값으로 나눈 비를 원하는 정확도 만큼의 2의 거듭제곱만큼 곱한다.
    #lambda_min_tilde = np.abs(lambda_min * 2**(n_l-neg_vals)/ lambda_max)
    #왜 -1? scaling 값이 eigenvalue의 최소값을 넘지 않도록 하기 위해서 작은 수를 빼준다?
    lambda_min_tilde = np.abs(lambda_min * (2**(n_l-neg_vals) - 1) / lambda_max)
    
    # floating point precision can cause problems
    if np.abs(lambda_min_tilde - 1) < 1e-7:
        lambda_min_tilde = 1
    
    #lambda_min_tilde를 주어진 bit만큼의 이진수의 string으로 표현한다. 앞의 0b를 지우기 위해서 2이후의 값을 사용
    binstr = format(int(lambda_min_tilde), formatstr)[2:]

    #이진수로 표현된 값을 다시 0~1사이의 이진수의 소수점 표현으로 해석해서 다시 십진법의 소수점 값으로 바꾼다.
    lamb_min_rep = 0
    for i, char in enumerate(binstr):
        lamb_min_rep += int(char) / (2 ** (i + 1))
    return lamb_min_rep

def value_setter(A):
    #eigenvalue와 관련된 값들을 불러온다.
    lambda_max, lambda_min, neg_vals = get_eigenvalue(A)
    #condition number를 계산한다.
    kappa = np.linalg.cond(A)
    print("condition number:", kappa)
    #A의 크기로 부터 nb값을 계산한다.
    nb = int(np.log2(len(A[0])))
    #적절한 nl값을 계산한다. condition number가 커질 수록 더 많은 qubit이 필요하게 된다.
    #n_l = 6
    n_l = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals
    #scaling과 evolution time을 계산하기 위한 delta(eigenvalue의 최소값을 최대값으로 나눈 것을 이진 소수점으로 정확하게 표현되게 한 것)를 계산한다.
    #scaling은 eigenvalue의 최소값(여기서는 최대값으로 나눈 값, tilde)에 가까운 값을 가져야 하기 때문에 delta값을 scaling값으로 표현한다.
    delta = get_delta(n_l, lambda_min, lambda_max, neg_vals)
    
    """
    evolution_time은 2pi를 eigenvalue의 최대 값으로 나눈 값으로 설정하는 것이 적절하다.
    따라서 앞서 계산한 delta를 eigenvalue의 최소값으로 나누면 eigenvalue의 최대값으로 나눈 효과를 가진다.
    음의 eigenvalue를 가지는 경우 추가적으로 2로 나누어준다.
    """
    evolution_time = 2 * np.pi * delta / lambda_min / (2**neg_vals)
    return n_l, evolution_time, delta, neg_vals


if __name__ == "__main__":
    get_delta(6, 2, 3, True)