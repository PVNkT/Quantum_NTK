import neural_tangents as nt
from jax import random
import jax.numpy as np
import numpy as np2
from jax.numpy.linalg import inv

class make_kernel():
    def __init__(self, kernel_fn, cfg, data):
        #주어진 seed에 맞게 key를 만듬
        self.key = random.PRNGKey(cfg.seed) #[ 0 12]
        # label별 데이터 수의 최대 값
        self.k_threshold = cfg.k_threshold
        #kernel 함수에 batch를 적용
        kernel_fn = nt.batch(kernel_fn, batch_size=cfg.batch_size)
        # 훈련 데이터와 테스트 데이터를 나눈다.
        self.train_data, self.test_data = data
        # the sparsified kernel will be asymmetric, so we can't just use the built-in cholesky
        # hence, we evaluate k_*^T K^{-1} y manually
        #kernel에 train, test이미지를 적용하여 train, test 데이터에 대한 kernel을 만든다. 
        self.kernel_train = kernel_fn(self.train_data['image'], self.train_data['image'], 'ntk')
        self.kernel_test = kernel_fn(self.test_data['image'], self.train_data['image'], 'ntk')
        print(self.kernel_train.shape, self.kernel_test)
        #sparse kernel을 만든다.
        kernel_train_sparse = self.sparsify(self.kernel_train, self.kernel_mag)
        #kernel의 대각선 요소만 남겨 diagonal kernel을 만든다.
        #numpy diag는 주어진 행렬의 대각 성분만을 추출해 1차원 array로 반환한다.
        #numpy eye는 주어진 크기의 identity 행렬을 만들고 추가적인 인수가 주어진 경우 그만큼 대각 성분이 이동한 값을 주게 된다.
        self.kernel_train_identity = np.diag(self.kernel_train)*np.eye(self.kernel_train.shape[0])
        #spase kernel의 대각선 성분의 최댓값*4로 이루어진 대각 행렬
        conditioning = 4*np.amax(np.diag(kernel_train_sparse))*np.eye(len(kernel_train_sparse))
        #sparse kernel의 대각 성분을 conditioning 값만큼 더해준다.
        #condition number를 줄이기 위한 과정. 4는 임의로 정한 값?
        self.kernel_train_sparse = kernel_train_sparse + conditioning
        #test kernel을 normalize한다. (데이터의 수 만큼으로 데이터를 나누고 값이 -1~1사이의 값을 가지도록 만든다. 
        # 그 후 데이터의 수만큼을 다시 곱해준다.)
        self.kernel_test_normalized = self.k_threshold*self.normalize(self.kernel_test)

    #kernel을 데이터 수만큼으로 나눈 값을 -1~1 사이 값으로 바꾸고 그것을 제곱한다.
    #0~1사이의 값이 나오게 됨
    def kernel_mag(self, row):
        return self.normalize(row)**2

    #m을 k_thresh로 나누고 -1보다 작은 값은 -1로 1보다 큰 값은 1로 바꾼다.
    def normalize(self,m):
        return np.clip(m/self.k_threshold, -1, 1)
    
    # kernel을 받아서 확률 함수에 따라서 
    def sparsify(self, m, probability_function):
        #원하는 sparsity(0이 아닌 항이 얼마나 많을 지)
        target_sparsity = int(5*np.log(m.shape[1]))

        #주어진 matrix와 같은 모양의 0으로된 행렬
        out = np2.zeros(m.shape)
        #주어진 행렬을 array로 바꿈
        m2 = np2.array(m)

        #행렬의 크기에 대해서 iteration
        for i in range(len(m)):
            # sample the other indices based on the probability function
            #주어진 확률 함수에 행렬의 행과 k_thresh를 입력
            probs = probability_function(m[i])
            #확률의 형태가 되도록 나누어줌
            probs /= np.sum(probs)

            #키를 2개로 나눔
            key, p_key = random.split(self.key, 2)
            #랜덤하게 0이 되지 않을 위치를 정한다.
            nonzero_indices = random.choice(p_key, np.arange(len(m)), shape=(target_sparsity,), replace=False, p=probs)
            #i가 nonzero_indice에 포함되지 않을 경우 포함시킨다. (대각 성분은 0이 되지 않게 한다.)
            if i not in nonzero_indices:
                nonzero_indices = np.concatenate((nonzero_indices, np.array([i])))
            #특정 index를 0으로 만들 mask를 만든다.
            mask = np2.zeros(m.shape[1], dtype=bool)
            #0이 되지 않을 부분은 1로 바꾼다.
            mask[(tuple(nonzero_indices),)] = 1
            #주어진 행렬에 mask를 곱해서 일부를 0으로 바꾼다.
            row = m2[i] * mask
            #결과에 각 행을 추가한다.
            out[i] += row
        #sparse matrix를 출력
        return np.array(out)

    def calc_exact(self):
        #평균 계산 k_*^T K^{-1} y
        #정확한 kernel을 사용해 계산한 값
        mean = self.kernel_test @ inv(self.kernel_train) @ self.train_data['label']
        return mean

    def calc_sparse(self):
        #sparse 행렬을 사용한 값
        mean_sparse = self.kernel_test_normalized @ inv(self.kernel_train_sparse) @ self.train_data['label']
        return mean_sparse

    def calc_identity(self):
        #diagonal 행렬을 사용한 값
        mean_identity = self.kernel_test_normalized @ inv(self.kernel_train_identity) @ self.train_data['label']
        return mean_identity








