
from omegaconf import OmegaConf
import numpy as np
import data_process
import network
from kernel import make_kernel
from result import make_result
from utils import logging_default
import logging

# config들을 불러와 한데 모으고, 데이터를 정제하여 커널을 생성하는 통제 함수.
def main(cfg = OmegaConf.load("config/config.yaml")): #config.yaml을 불러와서 cfg에 할당.
    #----------------------------------------------------------#
    #|              1. data initailization stage               |
    #----------------------------------------------------------#
    '''
    사용하는 파일 : config.yaml, MNIST.yaml, sphere.yaml
    '''

    
    # MNIST와 sphere data중 어떤 것을 사용할 것인지를 결정
    # sphere data는 d차원 구 위에서 데이터를 표현함
    cfg.merge_with_cli()
    data_type = str(cfg.data) # cfg에 들어있는 config.yaml로부터 data라는 key로 value를 호출함.
    # data의 종류에 따라서 필요한 parameter들을 불러와서 cli명령과 합침
    model_params = OmegaConf.load(f"config/{data_type}.yaml") #data_type에 맞는 yaml파일을 호출함.
    # 호출한 data_type.yaml파일을 위의 config.yaml과 OmegaConf.merge를 이용하여 합친다.
    cfg = OmegaConf.merge(cfg, model_params)    
    cfg.merge_with_cli()
    with logging_default(cfg) as logger:
        log=logger
    
    log.info("Data Initialization completed")
    
    #----------------------------------------------------------#
    #|             2. Getting Processed Data stage             |
    #----------------------------------------------------------#
    '''
    사용하는 파일 : data_process.py
    '''
  
    # 데이터의 종류에 따라서 parameter들을 입력받아 훈련 데이터와 테스트 데이터를 만든다.
    data = getattr(data_process, data_type + "_data_process") 
    data_class = data(cfg)
    datas = data_class.processed_train, data_class.processed_test 
    log.info("Data Processing completed")

    #----------------------------------------------------------#
    #|        3. Making Kernels and its assessment             |
    #----------------------------------------------------------#   
    '''
    사용하는 파일 : network.py, kernel.py, result.py
    '''

    # init, apply, kernel 함수를 만듬
    # 주어진 데이터에 따라서 신경망을 만들고 그 신경망에 대응되는 kernel함수를 계산한다.
    # MNIST데이터는 CNN신경망을 사용하고 sphere data는 dense layer를 사용한다.
    # network.py에서 지정한 data_type에 해당하는 함수를 들고오고, cfg로 초기화를 실시한다.
    # self.fn = self.create_network()에 해당하므로, network_class.fn은 네트워크 함수들을 불러옴.
    network_class = getattr(network, data_type + "_network")(cfg)
    # network_class.fn에서 stax.serial(*layer)로 들어가는 경우에, init_fun, apply_fun, preprocess_kernel을 내놓는다.
    pred_fn, _, kernel_fn = network_class.fn #kernel의 구조에 대한 것은 여기서 결정되어 나타난다.
    # 위에서 구한 kernel의 구조를 기반으로 아래에서 kernel에 대한 numerical한 계산을 실시함.
    # 주어진 kernel 함수를 통해서 batch등을 설정하고 그에 따른 여러 종류의 kernel을 만든다.
    # 정확히 계산된 kernel, sparse 과정을 거친 kernel, 대각 성분만 남긴 kernel을 각각 계산한다.
    kernels = make_kernel(kernel_fn=kernel_fn, cfg=cfg, data=datas)
    
    log.info("Kernel making completed")
    sparse = cfg.sparse


    for sparsity in np.arange(*tuple(dict(sparse).values())[1:3]):
        #앞서 만든 Sparse kernel을 통해서 평균에 대한 계산을 진행한다.
        mean = kernels.calc_sparse(sparsity, log) #(256, 2) : MNIST의 shape가 나옴. 0일확률과 1일 확률이 출력됨.
        # 계산 결과를 통해서 kernel들의 예측값을 얻고 이를 통해서 정확도를 계산하고 결과를 저장한다.
        make_result(cfg, mean, datas[1]['label'], sparsity)
        log.info(f"sparsity: {sparsity} Data storing...")

if __name__ == "__main__":
    main()

