from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization, Activation, add, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class XCEPTION(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/xception'
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
        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(input_layer)
        x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        # Block 2
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        # Block 3
        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = add([x, residual])

        residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        # Block 4
        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)
 
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
        x = add([x, residual])

        # Block 5-12
        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x)
            x = add([x, residual])

        residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization(axis=channel_axis)(residual)

        # Block 13
        x = Activation('relu', name='block13_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = Activation('relu', name='block13_sepconv2_act')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
        x = add([x, residual])

        # Block 14
        x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = Activation('relu', name='block14_sepconv1_act')(x)
        x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False,name='block14_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = Activation('relu', name='block14_sepconv2_act')(x)

        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        output_layer = Dense(n_classes, activation='softmax', name='predictions')(x)

        # define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)

        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)
        
        self.callbacks = [model_checkpoint, tb_cb]
        return model

