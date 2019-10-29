import logging

from utils_datagen import DataGenerator, create_partition_label_dict_multi_class, create_partition_label_dict_binary
from models.cnn_2d import CNN2D
from models.cnn_3d import CNN3D
from models.vgg19 import VGG19
from models.vgg19_3d import VGG193D
from models.xception import XCEPTION
from models.xception_3d import XCEPTION3D
from models.mvcnn_xception import MVCNN_XCEPTION
from models.mvcnn_cnn import MVCNN_CNN

logging.basicConfig(level=logging.DEBUG)

CLASSIFY_BINARY = True
TRAIN = False
MODEL3D = False

# model selection
model_types = ['CNN2D', 'CNN3D', 'VGG19', 'VGG193D', 'XCEPTION', 'XCEPTION3D', 'MVCNN_XCEPTION', 'MVCNN_CNN']
model_type = model_types[7]

# Dataset selection
pickle_files = ['./data/rois_all_slices_2d.p', './data/rois_all_slices_3d.p', './data/rois_all_slices_inverse_3d.p',
                './data/rois_first_four_slices_2d.p', './data/rois_first_four_slices_2d_rotated.p',
                './data/rois_first_four_slices_2d_more_normal.p', './data/rois_first_four_slices_list_more_normal.p',
                './data/rois_first_four_slices_3d.p']
pickle_file = pickle_files[6]
n_slices = 4

if CLASSIFY_BINARY:
    n_classes = 2
    partition, labels, integer_mapping_label_dict, image_shape = create_partition_label_dict_binary(pickle_file)
    print(integer_mapping_label_dict)
else:
    n_classes = 4
    partition, labels, integer_mapping_label_dict, image_shape = create_partition_label_dict_multi_class(pickle_file)

img_width, img_height = image_shape[0], image_shape[1]
if MODEL3D:
    input_shape = (img_width, img_height, n_slices, 1)
else:
    input_shape = (img_width, img_height, 1)

# input_shape = (128,128,1)

if model_type == 'CNN2D':
    model = CNN2D('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'CNN3D':
    model = CNN3D('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'VGG19':
    model = VGG19('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'VGG193D':
    model = VGG193D('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'XCEPTION':
    model = XCEPTION('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'XCEPTION3D':
    model = XCEPTION3D('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'MVCNN_XCEPTION':
    model = MVCNN_XCEPTION('./models/saved_models/', input_shape, n_classes, True)
if model_type == 'MVCNN_CNN':
    model = MVCNN_CNN('./models/saved_models/', input_shape, n_classes, True)

# Parameters
n_train_samples = len(partition['train'])
n_test_samples = len(partition['test'])

batch_size = 32
epochs = 200
samples_per_train_epoch = n_train_samples // batch_size
samples_per_test_epoch = n_test_samples // batch_size

params = {'dim': input_shape,
          'batch_size': batch_size,
          'n_classes': n_classes,
          'shuffle': False}

# Generators
testing_generator = DataGenerator(pickle_file, partition['test'], labels, integer_mapping_label_dict, **params)
training_generator = DataGenerator(pickle_file, partition['train'], labels, integer_mapping_label_dict, **params)

if TRAIN:
    model.fit(training_generator, testing_generator, samples_per_train_epoch, samples_per_test_epoch, epochs)

else:
    model.predict(testing_generator, samples_per_test_epoch)
