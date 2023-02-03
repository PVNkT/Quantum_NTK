from contextlib import contextmanager
import os
import csv
import jax.numpy as np

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

