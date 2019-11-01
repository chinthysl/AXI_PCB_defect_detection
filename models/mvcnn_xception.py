import keras
import os

from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Dense, concatenate, Maximum
from keras.layers import Dropout, Input, BatchNormalization, Activation, add, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
import keras.backend as backend
import keras.utils as keras_utils

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class MVCNN_XCEPTION(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/mvcnn_xception'
        chk_n_mkdir(self.output_directory)
        self.model = self.build_model(input_shape, n_classes)
        if verbose:
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + '/model_init.hdf5')

    def xception(self, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000):

        TF_WEIGHTS_PATH = (
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/'
            'xception_weights_tf_dim_ordering_tf_kernels.h5')
        TF_WEIGHTS_PATH_NO_TOP = (
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/'
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `imagenet` '
                             '(pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                             ' as true, `classes` should be 1000')

        # Determine proper input shape
        input_shape = (299, 299, 3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        channel_axis = -1

        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x, residual])

        residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x, residual])

        for i in range(8):
            residual = x

            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization(axis=channel_axis)(x)

            x = add([x, residual])

        residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x, residual])

        x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        if include_top:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = keras_utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x)

        # Load weights.
        if weights == 'imagenet':
            if include_top:
                weights_path = keras_utils.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels.h5',
                    TF_WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
            else:
                weights_path = keras_utils.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    TF_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b0042744bf5b25fce3cb969f33bebb97')
            model.load_weights(weights_path)
            if backend.backend() == 'theano':
                keras_utils.convert_all_kernels_in_model(model)
        elif weights is not None:
            model.load_weights(weights)

        return model

    def build_model(self, input_shape, n_classes):
        xception_1 = self.xception(include_top=False, pooling='max')
        for layer in xception_1.layers:
            layer.trainable = False
        input_1 = xception_1.input
        output_1 = xception_1.output
        xception_2 = self.xception(include_top=False, pooling='max')
        for layer in xception_2.layers:
            layer.trainable = False
        input_2 = xception_2.input
        output_2 = xception_2.output
        xception_3 = self.xception(include_top=False, pooling='max')
        for layer in xception_3.layers:
            layer.trainable = False
        input_3 = xception_3.input
        output_3 = xception_3.output
        xception_4 = self.xception(include_top=False, pooling='max')
        for layer in xception_4.layers:
            layer.trainable = False
        input_4 = xception_4.input
        output_4 = xception_4.output

        concat_layer = Maximum()([output_1, output_2, output_3, output_4])
        concat_layer.trainable = False
        # concat_layer = Dropout(0.25)(concat_layer)
        # dense_layer1 = Dense(units=1024, activation='relu')(concat_layer)
        dense_layer1 = Dropout(0.5)(concat_layer)
        output_layer = Dense(n_classes, activation='softmax', name='predictions')(dense_layer1)

        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=[output_layer])
        model.summary()
        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.01), metrics=['acc'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)
        
        self.callbacks = [model_checkpoint, tb_cb]
        return model


