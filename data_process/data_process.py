import jax.numpy as np
from jax import random
import tensorflow_datasets as tfds
from typing import List, Union

#MNIST data를 적절한 형태로 변형시키는 class
class MNIST_data_process:
    def __init__(self, cfg):
        #tensorflow dataset에서 MNIST 데이터를 최대크기의 batch 형태로 불러옴
        self.ds = tfds.as_numpy(tfds.load('mnist:3.*.*', batch_size=-1))
        #지정된 parameter들을 저장
        #어떤 숫자에 해당하는 데이터를 사용할 지 결정
        self.selection = cfg.selection
        # 사용할 데이터의 수 (class의 크기)
        self.k_threshold = cfg.k_threshold
        #랜덤 변수를 만들 seed
        seed = cfg.seed
        #주어진 seed를 통해서 랜덤한 key만듬
        self.key = random.PRNGKey(seed) #[ 0 12]

        #불러온 데이터를 train data와 test data로 나눔
        self.train_set = self.ds["train"]
        self.test_set = self.ds["test"]

        #데이터들에 대한 처리를 진행
        self.processed_train = self.process_data(self.train_set, self.selection, class_size = self.k_threshold)
        self.processed_test = self.process_data(self.test_set, self.selection, class_size = self.k_threshold)
        
    # 주어진 데이터를 기반으로 주어진 수만큼의 데이터로 나눈다.
    def process_data(self, data_chunk, selection, class_size=None, shuffle=True):
        # one-hot encode the labels and normalize the data.
        #데이터에서 image와 label에 해당되는 데이터만을 골라서 변수로 설정한다.
        image, label = data_chunk['image'], data_chunk['label']
        #라벨의 수를 계산한다.
        n_labels = len(selection)

        # pick two labels
        #선택한 label들에 해당되는 index들을 골라서 저장한다.
        indices = np.where((label == selection[0]) | (label == selection[1]))[0]
        #주어진 PRNG key를 두 개로 나눈다. 
        key, i_key = random.split(self.key, 2)
        #나누어진 key를 사용해서 index들의 array를 섞고 한 줄로 바꾼다. 
        indices = random.permutation(i_key, indices).reshape(1, -1)
        #0에 해당하는 label은 True, 1에 해당되는 label은 False로 하는 Tuple 데이터를 만든다. 
        label = (label[tuple(indices)] == selection[0])
        
        # balance if no class size is specified or class size too large
        #각 label이 몇 개씩 있는 지를 확인하고 그중 작은 값을 최대 class 크기로 정한다. 
        max_class_size = np.amin(np.unique(label, return_counts=True)[1])
        #class 크기가 지정되있지 않거나 최대 크기보다 클경우 max_class_size로 class 사이즈를 지정하고 출력한다.
        if (class_size is None) or class_size > max_class_size:
            class_size = max_class_size
            print('class_size', class_size)

        # select first class_size examples of each class
        #index를 새롭게 정의
        new_indices = []
        #선택한 label들의 종류들에 대해서 iteration
        for i in range(n_labels):
            #라벨이 일치하는 것만 찾음
            class_examples = np.where(label == i)[0]
            #크기를 class_size만큼으로 자르고 리스트로 바꾸어 new_indices 리스트에 추가
            new_indices += class_examples[:class_size].tolist()
        #랜덤 키를 나누어 다시 생성
        key, j_key = random.split(key, 2)
        #데이터의 셔플 여부 결정
        if shuffle:
            #인덱스를 다시 섞고 한 줄 형태로 바꾼다.
            new_indices = random.permutation(j_key, np.array(new_indices)).reshape(1, -1)
        else:
            #인덱스의 형태만 한 줄 형태로 바꾼다.
            new_indices = np.array(new_indices).reshape(1, -1)

        #새로운 index에 대응되는 label을 만듬
        label = label[tuple(new_indices)].astype(np.int64)
        #eye함수로 2*2 정방행렬을 만들고 이를 통해 label을 확률 분포의 형태로 만든다.
        # 0인 경우 [1 0], 1인 경우 [0 1]  
        label = np.eye(2)[label]

        #이미지 데이터를 다시 정리
        #앞서 정한 필요한 index의 image만 골라서 저장
        image = image[tuple(indices)][tuple(new_indices)]
        #image를 이미지 평균 값을 빼고 표준 편차로 나누어 정리한다.
        image = (image - np.mean(image)) / np.std(image)
        #이미지의 평균 값을 구하고 그것으로 나누어 이미지를 정규화한다.
        norm = np.sqrt(np.sum(image**2, axis=(1, 2, 3)))
        image /= norm[:, np.newaxis, np.newaxis, np.newaxis]

        #image와 label을 dictionary의 형태로 반환한다.
        return {'image': image, 'label': label}

class sphere_data_process:
    def __init__(self, cfg):
        self.noise_scale = 5e-2
        seed = cfg.seed
        key = random.PRNGKey(seed) #[ 0 12]
        self.key, self.x_key, self.y_key = random.split(key, 3)
        data_parameter = cfg.data_parameter
        self.processed_train, self.processed_test = self.process_data(**data_parameter)
    def target_fn(self, x):
        #주어진 데이터의 길이만큼의 array를 준비
        out = np.zeros(len(x))
        #주어진 데이터를 사용해서 sin(3/4*pi*n*x_n)을 모든 n에 대해서 더한다.
        for i in range(0, x.shape[1]):
            out += np.sin((i+1)*np.pi*x[:, i]*0.75)
        #출력 형태로 바꾸어 반환
        return np.reshape(out, (-1, 1))

    # train, test data를 만든다. 안쓰는듯
    def process_data(self, N, test_points=64, d=None, rand=False, rand_train=False):
        #전역 변수 key를 불러온다.
        #d값 설정
        if d is None:
            d = 3
        #random key를 옵션에 맞게 생성한다.
        if rand_train:
            key, train_x_key, train_y_key = random.split(self.key, 3)
        else:
            train_x_key = self.x_key
            train_y_key = self.y_key
        # noraml distridution의 랜덤 값을 N, d의 형태로 불러옴
        train_xs = random.normal(train_x_key, (N, d))
        # train_xs를 noramlize한다.
        # norm을 구하면서 차원이 하나 줄어든 것을 새로운 axis를 추가하고 d번만큼 반복하게 하여 차원을 맞춘다.
        norms = np.sqrt(np.sum(train_xs**2, axis=1))
        train_xs = train_xs / np.repeat(norms[:, np.newaxis], d, axis=1)

        # target_fn 함수를 통해서 train data를 변형시킴
        train_ys = self.target_fn(train_xs)
        # train data에 normal한 noise를 준다.
        train_ys += self.noise_scale * random.normal(train_y_key, (N, 1))
        # train_xs와 train_ys의 부호를 합쳐서 train 데이터로 저장
        train = (train_xs, np.sign(train_ys))

        if rand:
            #random한 normalize된 test 데이터를 만듬 
            key, test_x_key, test_y_key = random.split(key, 3)
            test_xs = random.normal(test_x_key, (test_points, d))
            norms = np.sqrt(np.sum(test_xs**2, axis=1))
            test_xs = test_xs / np.repeat(norms[:, np.newaxis], d, axis=1)
        else:
        # query points on a single path on the sphere surface
            # sin, cos 함수를 사용해서 test data를 만듬
            t = np.linspace(0, 2*np.pi, test_points)
            test_x_0 = np.reshape(np.sin(t), (-1, 1))
            test_x_1 = np.reshape(np.cos(t), (-1, 1))
            test_xs = np.concatenate((test_x_0, test_x_1, np.zeros((test_points, d-2))), axis=1)

        # target_fn 함수를 통해서 test_ys를 만듬
        test_ys = self.target_fn(test_xs)

        # test_xs와 test_ys의 부호를 tuple로 합친다.
        test = (test_xs, np.sign(test_ys))

        # train과 test 데이터를 반환
        return train, test