import jax.numpy as np
from neural_tangents import stax
import functools
from jax import random, jit
from jax.scipy.special import erf

# MNIST 데이터의 분석을 위해서 사용될 CNN구조를 만든다.
class MNIST_network:
    def __init__(self, cfg) -> None:
        # 신경망의 깊이를 얼마나 할지 불러온다.
        depth = cfg.depth
        # 깊이가 4인 경우 2,1,1의 형태로 나누고 아닌경우 3단계로 균등하게 나눈다.
        if depth == 4:
            self.depths = [2, 1, 1]
        else:
            self.depths = [depth//3, depth//3, depth//3]
        # output channel의 수, NTK에서는 무시된다.
        self.width = 1
        #주어진 조건으로 CNN 신경망을 만든다.
        self.fn = self.create_network()
    
    # MNIST neural network
    def create_network(self, W_std=np.sqrt(2.0), b_std=0.):
        #activation function으로 Relu함수를 사용
        activation_fn = stax.Relu()
        #convolution layer를 저장하는 list
        layers = []
        # functools.partial은 함수의 일부 입력값이 채워진 다른 함수를 만드는데 사용된다.
        #stax의 conv함수에서 W_std, b_std, padding값을 고정한 함수를 만든다.
        conv = functools.partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')
        
        #convolution과 activation function, average pooling과정을 나눈 depth만큼 추가하고 마지막의 flatten과정을 추가한다. 
        layers += [conv(self.width, (3, 3)), activation_fn] * self.depths[0]
        layers += [stax.AvgPool((2, 2), strides=(2, 2))]
        layers += [conv(self.width, (3, 3)), activation_fn] * self.depths[1]
        layers += [stax.AvgPool((2, 2), strides=(2, 2))]
        layers += [conv(self.width, (3, 3)), activation_fn] * self.depths[2]
        layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 2
        layers += [stax.Flatten(), stax.Dense(2, W_std, b_std)]

        #layers 내의 각 과정을 연속적으로 적용해서 반환한다.
        return stax.serial(*layers)

class sphere_network:
    def __init__(self,cfg) -> None:
        seed = cfg.seed
        L = cfg.L
        self.key = random.PRNGKey(seed) #[ 0 12]
        #erf에 대한 var값을 구한다. erf의 평균이 0이기 때문에 제곱 평균이 분산과 같다.
        self.scale = self.calc_var(erf, samples=10**8)
        #erf을 normalize하고 그에 대한 mu를 구한다. (기댓값?)
        self.mu = self.calc_mu(lambda x: np.sqrt(1/self.scale)*erf(x), samples=10**8)
        self.fn = self.create_network(L)
    # sphere neural network
    #주어진 함수에 normal distribution으로 샘플링한 값들의 제곱평균을 구한다.
    def calc_var(self, s, samples=10000):
        key, my_key = random.split(self.key)
        x = random.normal(my_key, (samples,))
        return np.mean(s(x)**2)
    # 주어진 함수에 normal distribution으로 샘플링한 값을 대입하고 그 값을 곱한 것의 평균을 제곱해서 1에서 뺀다. (?)
    def calc_mu(self, s, samples=10000):
        key, my_key = random.split(self.key)
        x = random.normal(my_key, (samples,))
        return 1 - np.mean(x*s(x))**2

    def create_network(self, L):
        #layer를 저장할 list
        layers = []

        #주어진 만큼 weight의 표준편차가 erf의 표준편차가 되도록하는 dense layer를 만들고 activation function으로 erf을 사용한다.
        for i in range(L):
            layers.append(stax.Dense(512, W_std=np.sqrt(1/self.scale), b_std=0.0))
            layers.append(stax.Erf())

        #앞서 만든 layer들에 출력 차원이 1이 되도록하는 dense layer를 만들고 그에 해당하는 init, apply, kernel 함수를 반환한다.
        init_fn, apply_fn, kernel_fn = stax.serial(
        *layers,
        stax.Dense(1, W_std=np.sqrt(1/self.scale), b_std=0.0)
        )

        #최적화된 compile방법을 적용
        apply_fn = jit(apply_fn)
        kernel_fn = jit(kernel_fn, static_argnums=(2,))

        #함수들을 반환
        return init_fn, apply_fn, kernel_fn

if __name__ == "__main__":
    import jax
    print(jax.default_backend())