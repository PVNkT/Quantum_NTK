import jax.numpy as np


class make_result:
    def __init__(self, cfg, means, labels):
        self.mean, self.mean_sparse, self.mean_identity = means
        self.cfg =cfg
        self.labels =labels
        acc = np.sum(self.classify(self.mean) == labels[:, 0])/len(labels)
        acc_sparse = np.sum(self.classify(self.mean_sparse) == labels[:, 0])/len(labels)
        acc_identity = np.sum(self.classify(self.mean_identity) == labels[:, 0])/len(labels)
        print('Exact classification accuracy:', acc)
        print('Sparse classification accuracy:', acc_sparse)
        print('Identity classification accuracy:', acc_identity)
        self.save_result()


    def path(self):
        cfg = self.cfg
        seed = cfg.seed
        selection = cfg.selection
        depth = cfg.depth
        threshold = cfg.k_threshold
        path_str = './output/mnist_seed_' + str(seed) + '_select_' + str(selection) + '_depth_' + str(cfg.depth) + '_data_' + str(cfg.k_threshold) + '_trial_' + str(cfg.trial)
        return path_str

    def classify(self,ntk_mean):
        # find classification threshold given balanced output
        # 0으로 추측된 경우에서 1로 추측한 경우를 뺀다.
        ntk_mean = ntk_mean[:, 0] - ntk_mean[:, 1]
        # 전체 결과의 중간값을 구한다.
        thresh = np.median(ntk_mean)
        # 중간값보다 큰 값은 1, 작은 값은 0으로 추측한다.
        out = (np.sign(ntk_mean - thresh).flatten() + 1) / 2
        return out

    def save_result(self):
        prefix = self.path()
        np.save(prefix + 'labels.npy', self.labels)
        np.save(prefix + 'exact.npy', self.mean)
        np.save(prefix + 'sparse.npy', self.mean_sparse)
        np.save(prefix + 'identity.npy', self.mean_identity)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("Quantum_NTK/config.yaml")
    result = make_result
    path = result(cfg, means=(np.array([1,2]),np.array([1,2]), np.array([1,2])), labels = np.array([0,1]))
    print(path)


