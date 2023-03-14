from contextlib import contextmanager
import os
import csv
import jax.numpy as np
import logging

@contextmanager
def csv_append(path_str):
    dir_name = os.path.dirname(path_str)
    New = True
    if os.path.isfile(path_str):
        New = False

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    
    f = open(path_str,  "a", newline='')
    wr = csv.writer(f)

    if New:
        wr.writerow(['sparsity','accuracy'])
        
    yield wr

    f.close()

@contextmanager
def npy_save(path_str, arr):
    dir_name = os.path.dirname(path_str)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    

    yield np.save(path_str, arr)

@contextmanager
def logging_default(cfg):
    # set path
    path_str = f"log/{cfg.selection}/{cfg.sparse.method}"
    # make path if it doesn't exit
    if not os.path.isdir(path_str):
        os.makedirs(path_str)
    logger = logging.getLogger("ntk_process")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # saving option
    if cfg.save.log:
        file_handler = logging.FileHandler(filename=path_str+f"/log_{cfg.seed}.log")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        pass
    # for console displaying
    #console_handler = logging.StreamHandler()
    #console_handler.setLevel(logging.INFO)
    #console_handler.setFormatter(formatter)
    #logger.addHandler(console_handler)
    
    yield logger




