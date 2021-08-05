import os, random, shutil


def chk_n_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_file(file_dir, target_dir, cls, rate=0.1):
    file_paths = os.listdir(file_dir)
    file_num = len(file_paths)
    pick_num = int(file_num * rate)
    random.seed(10)  # pls don't change this seed number
    test_sample = random.sample(file_paths, pick_num)
    # train_sample = list(set(file_paths) - set(test_sample))
    for name in file_paths:
        if name in test_sample:
            shutil.copyfile(file_dir + name, target_dir + "test/" + cls + "/" + name)
        else:
            shutil.copyfile(file_dir + name, target_dir + "train/" + cls + "/" + name)
    return


def split_data(DATA_DIR, TARGET_DIR, rate):
    chk_n_mkdir(TARGET_DIR + "test/" + "short/")
    chk_n_mkdir(TARGET_DIR + "test/" + "insufficient/")
    chk_n_mkdir(TARGET_DIR + "test/" + "missing/")
    chk_n_mkdir(TARGET_DIR + "test/" + "normal/")

    chk_n_mkdir(TARGET_DIR + "train/" + "short/")
    chk_n_mkdir(TARGET_DIR + "train/" + "insufficient/")
    chk_n_mkdir(TARGET_DIR + "train/" + "missing/")
    chk_n_mkdir(TARGET_DIR + "train/" + "normal/")

    classes = os.listdir(DATA_DIR)
    for cls in classes:
        if cls == "insufficient":
            copy_file(DATA_DIR + cls + '/', TARGET_DIR, cls, rate=rate)
        elif cls == "short":
            copy_file(DATA_DIR + cls + '/', TARGET_DIR, cls, rate=rate)
        elif cls == "missing":
            copy_file(DATA_DIR + cls + '/', TARGET_DIR, cls, rate=rate)
        elif cls == "normal":
            copy_file(DATA_DIR + cls + '/', TARGET_DIR, cls, rate=rate)


"""
Notes:
    After you generate the dataset(not splited) using the create_dataset.py,then you just need change 
    the path of your own dataset directory, DATA_DIR and TARGET_DIR, in the code and run the code.
"""

if __name__ == "__main__":
    DATA_DIR = "data/roi_concatenated_channelwise/"
    TARGET_DIR = "data/roi_concatenated_channelwise_splited/"
    split_data(DATA_DIR, TARGET_DIR, 0.2)
