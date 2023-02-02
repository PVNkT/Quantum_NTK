
from omegaconf import OmegaConf

import data_process
import network
from kernel import make_kernel
from result import make_result




def main(cfg = OmegaConf.load("config/config.yaml")):
    #MNIST와 sphere data중 어떤 것을 사용할 것인지를 결정
    #sphere data는 d차원 구 위에서 데이터를 표현함
    cfg.merge_with_cli()
    data_type = str(cfg.data)
    #data의 종류에 따라서 필요한 parameter들을 불러와서 cli명령과 합침
    model_params = OmegaConf.load(f"config/{data_type}.yaml")
    cfg = OmegaConf.merge(cfg, model_params)    
    cfg.merge_with_cli()
    
    #데이터의 종류에 따라서 parameter들을 입력받아 훈련 데이터와 테스트 데이터를 만든다.
    data = getattr(data_process, data_type + "_data_process") 
    data_class = data(cfg)
    datas = data_class.processed_train, data_class.processed_test 

    #init, apply, kernel 함수를 만듬
    #주어진 데이터에 따라서 신경망을 만들고 그 신경망에 대응되는 kernel함수를 계산한다.
    #MNIST데이터는 CNN신경망을 사용하고 sphere data는 dense layer를 사용한다.
    network_class = getattr(network, data_type + "_network")(cfg)
    pred_fn, _, kernel_fn = network_class.fn

    #주어진 kernel 함수를 통해서 batch등을 설정하고 그에 따른 여러 종류의 kernel을 만든다.
    #정확히 계산된 kernel, sparse 과정을 거친 kernel, 대각 성분만 남긴 kernel을 각각 계산한다.
    kernels = make_kernel(kernel_fn=kernel_fn, cfg=cfg, data=datas)
    #앞서 만든 kernel을 통해서 평균에 대한 계산을 진행한다.
    mean = kernels.calc_sparse()

    # 계산 결과를 통해서 kernel들의 예측값을 얻고 이를 통해서 정확도를 계산하고 결과를 저장한다.
    make_result(cfg, mean, datas[1]['label'])




if __name__ == "__main__":
    main()

