import logging
import pickle

from utils_datagen import DataGeneratorIndividual
from models.mvcnn_xception import MVCNN_XCEPTION

logging.basicConfig(level=logging.DEBUG)

CLASSIFY_BINARY = True
TRAIN = False

# model selection
model_types = ['MVCNN_XCEPTION']
model_type = model_types[0]
n_slices = 4

if CLASSIFY_BINARY:
    n_classes = 2
    pickle_file_name = './data/joints_4slices/details_list.p'
    with open(pickle_file_name, 'rb') as handle:
        details_list = pickle.load(handle)
    partition, temp_labels, temp_integer_mapping_label_dict, image_shape = details_list[0], details_list[1], details_list[2], details_list[3]
    labels = {}
    for image_name, value in temp_labels.items():
        if value == 3:
            new_value = 1
        else:
            new_value = 0
        labels[image_name] = new_value
    integer_mapping_label_dict = {'defect': 0, 'normal': 1}
else:
    n_classes = 4
    pickle_file_name = './data/joints_4slices/details_list.p'
    with open(pickle_file_name, 'rb') as handle:
        details_list = pickle.load(handle)
    partition, labels, integer_mapping_label_dict, image_shape = details_list[0], details_list[1], details_list[2], details_list[3]

img_width, img_height = image_shape[0], image_shape[1]
input_shape = image_shape

if model_type == 'MVCNN_XCEPTION':
    model = MVCNN_XCEPTION('./models/saved_models/', input_shape, n_classes, True)

# Parameters
n_train_samples = len(partition['train'])
n_test_samples = len(partition['test'])

batch_size = 32
epochs = 1000
samples_per_train_epoch = n_train_samples // batch_size
samples_per_test_epoch = n_test_samples // batch_size
 
params = {'dim': input_shape,
          'batch_size': batch_size,
          'n_classes': n_classes,
          'shuffle': False}


# Generators
testing_generator = DataGeneratorIndividual('./data/joints_4slices/', partition['test'], labels, integer_mapping_label_dict, **params)
training_generator = DataGeneratorIndividual('./data/joints_4slices/', partition['train'], labels, integer_mapping_label_dict,**params)


if TRAIN:
    model.fit(training_generator, testing_generator, samples_per_train_epoch, samples_per_test_epoch, epochs)

else:
    model.predict(testing_generator, samples_per_test_epoch)
