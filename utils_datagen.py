import numpy as np
import keras
import os
import tensorflow as tf
from keras.callbacks import TensorBoard
import pickle
import random
import cv2


def create_partition_label_dict_multi_class(pickle_file):
    with open(pickle_file, 'rb') as handle:
        image_dict = pickle.load(handle)

    train_key_list = []
    test_key_list = []
    partition = {'train': train_key_list, 'test': test_key_list}

    classes = ['missing', 'insufficient', 'short', 'normal']
    integer_mapping_dict = {x: i for i, x in enumerate(classes)}
    missing_key_list = []
    insufficient_key_list = []
    short_key_list = []
    normal_key_list = []

    labels = {}
    for image_key in image_dict.keys():
        label = image_dict[image_key][1]
        labels[image_key] = integer_mapping_dict[label]
        if label == 'missing':
            missing_key_list.append(image_key)
        elif label == 'insufficient':
            insufficient_key_list.append(image_key)
        elif label == 'short':
            short_key_list.append(image_key)
        elif label == 'normal':
            normal_key_list.append(image_key)

    for key_list in [missing_key_list, insufficient_key_list, short_key_list, normal_key_list]:
        num_train = int(len(key_list) * 0.9)
        num_images = len(key_list)
        train_indices = random.sample(range(num_images), num_train)
        test_indices = []
        for index in range(num_images):
            if index not in train_indices:
                test_indices.append(index)
        
        for index in train_indices:
            train_key_list.append(key_list[index])
            
        for index in test_indices:
            test_key_list.append(key_list[index])

    if isinstance(next(iter(image_dict.values()))[0], list):
        image_shape = next(iter(image_dict.values()))[0][0].shape
        print('slices list selected, image shape:', image_shape)
    else:
        image_shape = next(iter(image_dict.values()))[0].shape

    return partition, labels, integer_mapping_dict, image_shape


def create_partition_label_dict_binary(pickle_file):
    with open(pickle_file, 'rb') as handle:
        image_dict = pickle.load(handle)

    train_key_list = []
    test_key_list = []
    partition = {'train': train_key_list, 'test': test_key_list}

    integer_mapping_dict = {'missing': 0, 'insufficient': 0, 'short': 0, 'normal': 1}
    defective_key_list = []
    normal_key_list = []

    labels = {}
    for image_key in image_dict.keys():
        label = image_dict[image_key][1]
        labels[image_key] = integer_mapping_dict[label]
        if label == 'missing' or label == 'insufficient' or label == 'short':
            defective_key_list.append(image_key)
        elif label == 'normal':
            normal_key_list.append(image_key)

    for key_list in [defective_key_list, normal_key_list]:
        num_train = int(len(key_list) * 0.9)
        num_images = len(key_list)
        train_indices = random.sample(range(num_images), num_train)
        test_indices = []
        for index in range(num_images):
            if index not in train_indices:
                test_indices.append(index)

        for index in train_indices:
            train_key_list.append(key_list[index])

        for index in test_indices:
            test_key_list.append(key_list[index])

    if isinstance(next(iter(image_dict.values()))[0], list):
        image_shape = next(iter(image_dict.values()))[0][0].shape
        print('slices list selected, image shape:', image_shape)
    else:
        image_shape = next(iter(image_dict.values()))[0].shape

    integer_mapping_dict = {'defect': 0, 'normal': 1}
    return partition, labels, integer_mapping_dict, image_shape


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pickle_file, image_keys, labels, integer_mapping_dict, batch_size, dim, n_classes, shuffle=True):
        'Initialization'
        with open(pickle_file, 'rb') as handle:
            image_dict = pickle.load(handle) 
        self.image_dict = image_dict
        self.labels = labels
        self.image_keys = image_keys
        self.integer_mapping_dict = integer_mapping_dict
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def gen_class_labels(self):
        classes = []
        end_idx = len(self.image_keys) // self.batch_size * self.batch_size
        for i in self.image_keys[0:end_idx]:
            classes.append(self.labels[i])
        return classes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_keys) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_keys_temp = [self.image_keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_keys_temp)
        return X, y

        # X1, X2, X3, X4, y = self.__data_generation(image_keys_temp)
        # return [X1, X2, X3, X4], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_keys))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_keys_temp):
        # Generates data containing batch_size samples, X : (n_samples, *dim, n_channels)

        # # Initialization
        # X = np.empty((self.batch_size, *self.dim))
        # y = np.empty(self.batch_size, dtype=int)
        #
        # # Generate data
        # for i, ID in enumerate(image_keys_temp):
        #     # Store sample
        #     image = self.image_dict[ID][0]
        #     # if len(image.shape) == 2:
        #     #     image = np.expand_dims(image, axis=2)
        #     X[i, ] = image
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        #
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

        # # Initialization
        # X = np.empty((self.batch_size, *self.dim))
        # y = np.empty(self.batch_size, dtype=int)
        #
        # # Generate data
        # for i, ID in enumerate(image_keys_temp):
        #     # Store sample
        #     image = self.image_dict[ID][0]
        #     image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_AREA)
        #     image = np.expand_dims(image, axis=2)
        #     X[i, ] = image
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        #
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

        # # Initialization
        # X1 = np.empty((self.batch_size, 128, 128, 1))
        # X2 = np.empty((self.batch_size, 128, 128, 1))
        # X3 = np.empty((self.batch_size, 128, 128, 1))
        # X4 = np.empty((self.batch_size, 128, 128, 1))
        # y = np.empty(self.batch_size, dtype=int)
        #
        # # Generate data
        # for i, ID in enumerate(image_keys_temp):
        #     # Store sample
        #     images = self.image_dict[ID][0]
        #     # print(images[:,:,0,:].shape)
        #     image1 = images[:,:,0,:]
        #     image2 = images[:,:,1,:]
        #     image3 = images[:,:,2,:]
        #     image4 = images[:,:,3,:]
        #     X1[i,] = image1
        #     X2[i,] = image2
        #     X3[i,] = image3
        #     X4[i,] = image4
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        #
        # return [X1, X2, X3, X4], keras.utils.to_categorical(y, num_classes=self.n_classes)

        # Initialization
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))
        X3 = np.empty((self.batch_size, *self.dim))
        X4 = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(image_keys_temp):
            # Store sample
            slices_list = self.image_dict[ID][0]

            image1 = slices_list[0]
            image2 = slices_list[1]
            image3 = slices_list[2]
            image4 = slices_list[3]
            X1[i,] = image1
            X2[i,] = image2
            X3[i,] = image3
            X4[i,] = image4

            # Store class
            y[i] = self.labels[ID]

        return [X1, X2, X3, X4], keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGeneratorIndividual(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pickle_file_dir, image_keys, labels, integer_mapping_dict, batch_size, dim, n_classes, shuffle=True):
        'Initialization'
        self.pickle_file_dir = pickle_file_dir
        self.labels = labels
        self.image_keys = image_keys
        self.integer_mapping_dict = integer_mapping_dict
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def gen_class_labels(self):
        classes = []
        end_idx = len(self.image_keys) // self.batch_size * self.batch_size
        for i in self.image_keys[0:end_idx]:
            classes.append(self.labels[i])
        return classes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_keys) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_keys_temp = [self.image_keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_keys_temp)
        return X, y

        # X1, X2, X3, X4, y = self.__data_generation(image_keys_temp)
        # return [X1, X2, X3, X4], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_keys))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_keys_temp):
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))
        X3 = np.empty((self.batch_size, *self.dim))
        X4 = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(image_keys_temp):
            # Store sample
            with open(self.pickle_file_dir + ID, 'rb') as handle:
                slices_list = pickle.load(handle)

            image1 = slices_list[0]
            image2 = slices_list[1]
            image3 = slices_list[2]
            image4 = slices_list[3]
            X1[i,] = image1
            X2[i,] = image2
            X3[i,] = image3
            X4[i,] = image4

            # Store class
            y[i] = self.labels[ID]

        return [X1, X2, X3, X4], keras.utils.to_categorical(y, num_classes=self.n_classes)


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log', **kwargs):
        self.log_dir = log_dir
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(self.log_dir, '/training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(self.log_dir, '/validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
