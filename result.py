import csv

import jax.numpy as np


class make_result:
    def __init__(self, cfg, means, labels):
        #여러 형태의 kernel에 대해 계산한 평균값을 불러옴
        self.mean = means
        #label들을 불러옴
        self.cfg = cfg
        self.labels =labels
        #평균을 기반으로 각 kernel로 계산한 값의 정확도를 계산함
        acc = np.sum(self.classify(self.mean) == labels[:, 0])/len(labels)
        self.sparse_mode = cfg.sparse.method
        # 계산된 정확도를 출력
        print(f'sparsity {cfg.sparse.sparsity} accuracy:', acc)
        f = open(f'accuracy/{self.sparse_mode}/{self.sparse_mode}_sparse_output_{cfg.seed}.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([cfg.sparse.sparsity, acc])
        f.close()
        # 계산 결과를 npy파일로 저장
        self.save_result()

    # 설정값을 기반으로 저장할 파일의 이름을 만드는 함수
    def path(self):
        cfg = self.cfg
        path_str = f'./output/{cfg.data}_method_{self.sparse_mode}_sparsity_{cfg.sparse.sparsity}_seed_{cfg.seed}_select_{cfg.selection}_depth_{cfg.depth}_data_{cfg.k_threshold}_trial_{cfg.trial}'
        return path_str
    
    # kernel을 통해서 얻은 평균값을 통해서 어떤 값으로 예측하였는지를 확인하는 함수
    def classify(self,ntk_mean):
        # find classification threshold given balanced output
        # 0으로 추측된 정도에서 1로 추측한 정도를 빼서 추측값을 구한다.
        ntk_mean = ntk_mean[:, 0] - ntk_mean[:, 1]
        # 테스트 데이터들에 대한 추측 값들의 중간값을 구한다.
        thresh = np.median(ntk_mean)
        # 중간값보다 큰 값은 1, 작은 값은 0으로 추측한다.
        out = (np.sign(ntk_mean - thresh).flatten() + 1) / 2
        return out

    # 계산된 평균값들을 npy형식으로 저장한다.
    def save_result(self):
        prefix = self.path()
        np.save(prefix + 'labels.npy', self.labels)
        np.save(prefix + '.npy', self.mean)



if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("Quantum_NTK/config.yaml")
    result = make_result
    path = result(cfg, means=(np.array([1,2]),np.array([1,2]), np.array([1,2])), labels = np.array([0,1]))
    print(path)


