from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization, Activation, add, GlobalAveragePooling3D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class XCEPTION3D(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/xception_3d'
        chk_n_mkdir(self.output_directory)
        self.model = self.build_model(input_shape, n_classes)
        if verbose:
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + '/model_init.hdf5')

    def build_model(self, input_shape, n_classes):
        # input layer
        input_layer = Input(input_shape)
        channel_axis = -1  # channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

        # Block 1
        x = Conv3D(8, (3, 3, 3), use_bias=False, name='block1_conv1')(input_layer)
        x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv3D(8, (3, 3, 2), use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Conv3D(16, (1, 1, 1), strides=(2, 2, 1), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        # Block 2
        x = Conv3D(16, (3, 3, 1), padding='same', use_bias=False, name='block2_conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block2_conv1_bn')(x)

        x = MaxPooling3D((3, 3, 1), strides=(2, 2, 1), padding='same', name='block2_pool')(x)
        x = add([x, residual])

        residual = Conv3D(32, (1, 1, 1), strides=(2, 2, 1), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        # Block 3
        x = Activation('relu', name='block3_conv1_act')(x)
        x = Conv3D(32, (3, 3, 1), padding='same', use_bias=False, name='block3_conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block3_conv1_bn')(x)

        x = MaxPooling3D((3, 3, 1), strides=(2, 2, 1), padding='same', name='block3_pool')(x)
        x = add([x, residual])

        # Block 4
        x = Conv3D(64, (3, 3, 1), padding='same', use_bias=False, name='block4_conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block4_conv1_bn')(x)
        x = Activation('relu', name='block4_conv1_act')(x)

        # Classification block
        x = GlobalAveragePooling3D(name='avg_pool')(x)
        output_layer = Dense(n_classes, activation='softmax', name='predictions')(x)

        # ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
        # ## add dropouts to avoid overfitting / perform regularization
        # dense_layer1 = Dense(units=2048, activation='relu')(x)
        # dense_layer1 = Dropout(0.4)(dense_layer1)
        # dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
        # dense_layer2 = Dropout(0.4)(dense_layer2)
        # output_layer = Dense(units=n_classes, activation='softmax')(dense_layer2)

        # define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)

        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)
        
        self.callbacks = [model_checkpoint, tb_cb]
        return model
