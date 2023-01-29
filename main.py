
from omegaconf import OmegaConf

import data_process
import network
from kernel import make_kernel
from result import make_result




def main(cfg = OmegaConf.load("config.yaml")):
    data_type = str(cfg.data)
    
    model_params = OmegaConf.load(f"{data_type}.yaml")
    cfg = OmegaConf.merge(cfg, model_params)    
    cfg.merge_with_cli()
    
  
    data = getattr(data_process, data_type + "_data_process") 
    data_class = data(cfg)
    data = data_class.processed_train, data_class.processed_test
    
    train_data, test_data = data
    labels = test_data['label']
    #init, apply, kernel 함수를 만듬
    network_class = getattr(network, data_type + "_network")(cfg)
    pred_fn, _, kernel_fn = network_class.fn

    kernels = make_kernel(kernel_fn=kernel_fn, cfg=cfg, data=data)
    means = kernels.calc_mean()

    make_result(cfg, means, labels)



if __name__ == "__main__":
    main()

