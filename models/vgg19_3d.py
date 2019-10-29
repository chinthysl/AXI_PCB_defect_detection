from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class VGG193D(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/vgg19_3d'
        chk_n_mkdir(self.output_directory)
        self.model = self.build_model(input_shape, n_classes)
        if verbose:
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, n_classes):
        # input layer
        input_layer = Input(input_shape)
        # Block 1
        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPool3D((2, 2, 2), strides=(2, 2, 1), name='block1_pool')(x)

        # Block 2
        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPool3D((2, 2, 2), strides=(2, 2, 1), name='block2_pool')(x)

        # Block 3
        x = Conv3D(256, (3, 3, 2), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv3D(256, (3, 3, 2), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv3D(256, (3, 3, 2), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv3D(256, (3, 3, 2), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPool3D((2, 2, 2), strides=(2, 2, 1), name='block3_pool')(x)

        # Block 4
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), name='block4_pool')(x)

        # Block 5
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv3D(512, (3, 3, 1), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), name='block5_pool')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.4)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.4)(x)
        output_layer = Dense(n_classes, activation='softmax', name='predictions')(x)

        ## define the model with input layer and output layer
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

