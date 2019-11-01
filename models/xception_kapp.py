from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization, Activation, add, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import callbacks
from keras import models
from keras.applications import Xception

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class XCEPTION_APP(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/xception_kapp'
        chk_n_mkdir(self.output_directory)
        self.model = self.build_model(input_shape, n_classes)
        if verbose:
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + '/model_init.hdf5')

    def build_model(self, input_shape, n_classes):
        # Load the VGG model
        xception_conv = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze the layers except the last 4 layers
        for layer in xception_conv.layers:
            layer.trainable = False

        # Create the model
        model = models.Sequential()

        # Add the vgg convolutional base model
        model.add(xception_conv)
        # Add new layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax', name='predictions'))

        # define the model with input layer and output layer
        model.summary()
        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.01), metrics=['acc'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)

        self.callbacks = [model_checkpoint, tb_cb]
        return model

