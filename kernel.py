import neural_tangents as nt
from jax import random
import jax.numpy as np
import numpy as np2
from jax.numpy.linalg import inv
from scipy.linalg import block_diag
from omegaconf import OmegaConf

from hhl.solver import HHL_my
from utils import npy_save
import ipdb

#kernel 생성을 위한 class.
class make_kernel():
    def __init__(self, kernel_fn, cfg, data):
        '''
        <생성되는 커널들>
        1. self.kernel_train : 훈련데이터를 이용해 정확하게 계산된 커널
        2. self.kernel_test : 테스트 데이터를 이용해 정확하게 계산된 커널
        3. self.kernel_train_sparse  : Random Process를 통해 sparse Kernel을 생성.
        4. self.kernel_train_identity : Diagonal term들만 남아있음.
        5. self.kernel_test_normalized : 테스트 데이터를 이용해 정확하게 계산된 커널을 normalize한 커널.
        '''

        #주어진 seed에 맞게 key를 만듬
        self.seed = cfg.seed
        self.key = random.PRNGKey(self.seed) #[ 0 12]
        # label별 데이터 수의 최대 값
        self.k_threshold = cfg.k_threshold 
        # sparse method에는 origin, random, block이 존재.
        self.sparse_method = cfg.sparse.method
        #kernel 함수에 batch를 적용
        kernel_fn = nt.batch(kernel_fn, batch_size=cfg.batch_size)
        # 훈련 데이터와 테스트 데이터를 나눈다.
        self.train_data, self.test_data = data
        # the sparsified kernel will be asymmetric, so we can't just use the built-in cholesky
        # hence, we evaluate k_*^T K^{-1} y manually
        # kernel에 train, test이미지를 적용하여 train, test 데이터에 대한 kernel을 만든다. 
        self.kernel_train = kernel_fn(self.train_data['image'], self.train_data['image'], 'ntk')
        self.selection = cfg.selection
        self.kernel_test = kernel_fn(self.test_data['image'], self.train_data['image'], 'ntk')
        self.hhl = cfg.hhl
        with npy_save(f'/workspace/kernel/{self.selection}/{self.sparse_method}/{self.seed}/kernel_test.npy', self.kernel_test) as npy:
            npy

    def sparsifying(self, sparsity):
        # sparse kernel을 만든다.
        # sparsify라는 함수에 두번째 인자인 probability_function 위치에 kernel_mag라는 함수 자체를 넘겨주었음을 알아두자.
        # sparse한 행렬을 sparsify를 통해 생성하고 train데이터를 이용.
        #class를 저장하여 getattr로 이름에 맞는
        if self.sparse_method == "block":
            sparsity = 2**sparsity 
        self.sparsifying_class = sparsify(self.kernel_train, sparsity, self.key, self.k_threshold)
        self.sparsify = getattr(self.sparsifying_class, self.sparse_method)

    def normalize(self):
        # test kernel을 normalize한다. (데이터의 수 만큼으로 데이터를 나누고 값이 -1~1사이의 값을 가지도록 만든다. 
        # 그 후 데이터의 수만큼을 다시 곱해준다.)
        # sparsifying_class에는 tool이 상속되어져 있으므로 아래와 같은 방법으로 호출.
        normalize = getattr(self.sparsifying_class, "normalize")
        self.kernel_test_normalized = self.k_threshold*normalize(self.kernel_test)
        return None
    
    #kernel의 평균을 계산하기 위한 함수.
    def calc_exact(self):
        # 평균 계산 k_*^T K^{-1} y
        # 정확한 kernel을 사용해 계산한 값
        if self.hhl == False:
            mean = self.kernel_test @ inv(self.kernel_train) @ self.train_data['label']
        else :
            #ipdb.set_trace()
            mean = self.kernel_test @ HHL_my(self.kernel_train, self.train_data['label'], wrap = True, measurement = None)
        return mean
    
    #sparse kernel의 평균을 계산하기 위한 값. 
    def calc_sparse(self, sparsity):
        self.sparsifying(sparsity)
        self.normalize()
        # sparse 행렬을 사용한 값
        self.kernel_train_sparse = self.sparsify()
        with npy_save(f'/workspace/kernel/{self.selection}/{self.sparse_method}/{self.seed}/kernel_train_{sparsity}.npy', self.kernel_train_sparse) as npy:
            npy

        if self.hhl == False:
            #ipdb.set_trace()
            mean_sparse = self.kernel_test_normalized @ inv(self.kernel_train_sparse) @ self.train_data['label']
        else :
            '''
            calculate column by column
            this method is for binary classification. 
            for more class problem, this method must be revised!
            jax에서 제공하는 Devicearray를 이용하나, np.array로 바꿔서 작업하나 결과는 이상이 없음.
            다만 train_data의 형타입을 int로 맞춰야 작업이 돌아감.
            '''

            train_data = np2.array(self.train_data['label'],dtype=int) #self.train_data['label'][:,0]
            #ipdb.set_trace()
            col_1 = self.kernel_test_normalized @ HHL_my(self.kernel_train_sparse, train_data[:,0], wrap = True, measurement = None)
            col_2 = self.kernel_test_normalized @ HHL_my(self.kernel_train_sparse, train_data[:,1], wrap = True, measurement = None)
            #ipdb.set_trace()
            mean_sparse = np.vstack((col_1,col_2)).T
            print("check:",mean_sparse)
        return mean_sparse


#kernel의 잡다한 것들을 계산하기 위한 class.
class tools:
    def __init__(self, kernel, k_threshold):
        self.original_kernel = kernel
        self.k_threshold = k_threshold

    #kernel의 magnitude를 계산해주는 함수.
    def kernel_mag(self, row):
        '''
        kernel을 데이터 수만큼으로 나눈 값을 -1~1 사이 값으로 바꾸고 그것을 제곱한다.
        0~1사이의 값이 나오게 됨
        '''
        return self.normalize(row)**2

    #주어진 kernel 값을 normalize 해주는 함수
    def normalize(self,m):
        #m을 k_thresh로 나누고 -1보다 작은 값은 -1로 1보다 큰 값은 1로 바꾼다.
        return np.clip(m/self.k_threshold, -1, 1)

    #sparse kernel의 대각성분에 conditioning을 곱해주는 과정.
    def conditioning(self, kernel):
        # sparse kernel의 대각선 성분의 최댓값*4로 이루어진 대각 행렬
        conditioning = 4*np.amax(np.diag(kernel))*np.eye(kernel.shape[0])
        # sparse kernel의 대각 성분을 conditioning 값만큼 더해준다.
        # condition number를 줄이기 위한 과정. 4는 임의로 정한 값?
        return kernel + conditioning

#tools에 sparsification을 위한 함수들을 모아놓은 class.
class sparsify(tools): #tools를 상속받아옴
    def __init__(self, m, sparsity, key, k_threshold): #m자리에 test/train kernel을 넣었음
        self.key = key
        self.sparsity = sparsity
        self.k_threshold = k_threshold
        self.original_kernel = m

    # sparse matrix를 주어진 sparse 확률에 따라서 만들어주는 함수.
    def origin(self): #decipriated!

        #sparsify되기 전의 kernel
        m = self.original_kernel
        #원하는 sparsity(0이 아닌 항이 얼마나 많을 지)
        target_sparsity = int(self.sparsity*np.log(m.shape[1]))
        probability_function = self.kernel_mag
        #주어진 matrix와 같은 모양의 0으로된 행렬
        out = np2.zeros(m.shape)
        #주어진 kernel을 array로 바꿈
        m2 = np2.array(m)
        
        '''
        여기서 이용하는 sparsify의 원리는 다음과 같다.
        kernel의 한 행씩을 불러와서, for loop로 row by row로 연산을 실시함.
        0이 되지 않을 부분들을 random.choice를 이용하여 선택하고 
        그 index들을 추출하여 nonzero_indices에 할당한다.
        
        그 다음 diagonal한 성분들이 0이 되지 않게 하기 위해서,
        nonzero_indeices에 diagonal한 index를 추가한다.

        nonzero_indices들은 1 값을 갖고 그 외의 값들은 0을 갖는 행렬을 만들고 kernel과 곱한다.
        그러므로 zero인 indices들은 곱한 값이 0이 되므로 이와 같은 방식으로 sparsify를 실시한다.
        이후 out이라는 리스트에 해당 리스트를 추가하고 이후 이를 array로 바꾸어 반환한다.
        '''

        #행렬의 크기에 대해서 iteration
        for i in range(len(m)):
            # sample the other indices based on the probability function
            #주어진 확률 함수에 행렬의 행과 k_thresh를 입력
            #위에서 인자로 probability function으로 kernel_mag가 들어왔음.
            #kernel_mag함수에 test/train kernel의 i번째 행을 넣어서 kernel의 magnitude를 얻음.
            probs = probability_function(m[i])
            #확률의 형태가 되도록 나누어줌
            probs /= np.sum(probs)

            #키를 2개로 나눔
            key, p_key = random.split(self.key, 2)
            #랜덤하게 0이 되지 않을 위치를 정한다. (중복되지 않도록)
            #ipdb.set_trace()
            nonzero_indices = random.choice(p_key, np.arange(len(m)), shape=(target_sparsity,), replace=False, p=probs) #주어진 kernel element 값에 따라서 0이 될 확률이 정해짐.
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
        #sparse matrix
        sparse_matrix = np.array(out)
        #conditioning을 진행
        self.conditioned_matrix = self.conditioning(sparse_matrix)
        return self.conditioned_matrix
    
    # Random sparse kernel matrix를 만들어주는 함수.
    def random(self):
        '''
        random mask를 생성하여 kernel과 곱하는 형식으로 random sparse matrix를 구현하는 함수.
        여기서 mask는 random seed로 생성하여 만들어진 array에 
        sparsity보다 작으면 1 크면 1로 조건을 걸어서 array를 생성함.
        '''
        if self.sparsity>1:
            raise "probability should be smaller than 1"
        # 원래 형태의 kernel을 받아서 m에 저장함.
        m = self.original_kernel
        # key에 해당하는 seed 생성.
        np2.random.seed(self.key)
        # kernel과 같은 shape의 난수로 생성된 mask를 생성함.
        mask = np2.random.random(m.shape)
        # mask의 값이 self.sparsity보다 작다면 1로, 아니면 0으로 변경
        mask = np2.where(mask < self.sparsity, 1, 0)
        # 정방행렬인 이상, diagonal 성분의 개수는 행 혹은 열의 개수와 동일하다.
        # di에 행렬의 diagonal 성분의 인덱스를 저장함.
        # https://numpy.org/doc/stable/reference/generated/numpy.diag_indices.html
        di = np2.diag_indices(m.shape[0])
        # mask의 diagonal term들을 모두 1로 저장한다.
        mask[di] = 1
        #original_kernel의 값에 mask를 곱해서 대각성분을 제외하고 random하게 sparsity를 갖는 kernel을 생성.
        kernel = np.array(mask) * m
        self.conditioned_matirx = self.conditioning(kernel)
        return self.conditioned_matirx    
    
    #block diagonalization을 실시하는 과정
    def block(self):
        '''
        kernel matrix의 행의 개수를 sparsity로 나눈 값의 정수형을 취한다.
        취한 값을 i에 대하여 iteration을 실시하여, block을 지정한다.
        지정한 blocks를 이용하여 block_diag를 통하여 block_diagonalize를 실시한다.
        
        여기서 block_diag의 경우, block들에 대한 어레이들을 받아와서 blocks에 저장하고
        이를 diagonal한 position에 위치시킨 후에, 나머지 요소들에 0을 채우는 형식으로 block_diagonal을 실시.
        '''
        
        #kernel을 인자로 받아옴
        m = self.original_kernel
        #kernel은 정방행렬로, 행을 들고옴
        size = int(m.shape[0])
        #l에는 sparsity를 저장
        l = int(self.sparsity)

        #행의 개수가 sparsity로 나누어 떨어지지 않는 경우에 대한 error raise
        #자동화를 위해서 shape를 factorize해야함.
        if size % l != 0:
            raise f"size should be divided by sparsity! size{size}, sparsity{l}"

        #kernel matrix에서 block diagonal을 실시할 블럭을 지정함.
        
        blocks = [np2.array(m[i:i+l,i:i+l]) for i in range(int(size/l))]
        diag_block = block_diag(*blocks)
        self.conditioned_matrix = self.conditioning(np.array(diag_block))
        return self.conditioned_matrix

    #diagonal 항만 남겨서 kernel matrix를 만들어주는 함수.
    def diagonal(self):
        '''
        kernel의 대각선 요소만 남겨 diagonal kernel을 만든다.
        numpy diag는 주어진 행렬의 대각 성분만을 추출해 1차원 array로 반환한다.
        numpy eye는 주어진 크기의 identity 행렬을 만들고 추가적인 인수가 주어진 경우 그만큼 대각 성분이 이동한 값을 주게 된다.
        그러므로 diagonal element들만 빼놓고 모두 0이 되는 kernel을 나타낸다.
        '''
        N = int(self.original_kernel.shape[0])
        diagonal = np.zeros((N,N))
        diagonal += np.diag(np.diag(self.original_kernel))
        for i in range(1,self.sparsity+1):
            diagonal += np.diag(np.diag(self.original_kernel,k=i),k=i)
            diagonal += np.diag(np.diag(self.original_kernel,k=-i),k=-i)
        self.conditioned_matrix = self.conditioning(diagonal)
        return self.conditioned_matrix
    
    #kernel의 element들을 크기별로 나열하여 작은 순서대로 나열하여 순차적으로 0으로 만듦
    def threshold(self):
        l = self.sparsity
        m = self.original_kernel
        #element들의 크기순대로 인덱스를 가져오기
        ind = np2.unravel_index(m.argsort(axis=None), m.shape)
        #sparsity로 array slicing 실시
        ind = ind[0][:l], ind[1][:l]
        
        #mask 생성
        mask = np2.zeros(m.shape)
        #slicing한 인덱스들을 0으로 만듦
        mask[ind]=1
        #mask의 diagonal element들을 0으로 만듦
        mask[np2.diag_indices_from(mask)] = 0
        #original kernel에 mask를 곱함
        ths_kernel = np2.array(mask) * m
        #빠졌던 diagonal element들에 따로 저장해놓은 diagonal element들을 저장함.
        ths_kernel += np.diag(np.diag(self.original_kernel))
        return ths_kernel

    
if __name__=="__main__":
    cfg = OmegaConf.load("./config/MNIST.yaml")
    cfg.merge_with_cli()
    key = random.PRNGKey(12)
    k_threshold = cfg.k_threshold
    sparsity = 15
    #m = np2.ones((5,5))#(256,256)
    m = np2.random.rand(5,5)
    kernel = sparsify(m, sparsity, key, k_threshold).threshold()
    np.save('/workspace/thstestkernel',kernel)






