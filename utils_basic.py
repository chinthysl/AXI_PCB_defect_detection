import os


def chk_n_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

