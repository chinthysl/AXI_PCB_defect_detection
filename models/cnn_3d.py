from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Adadelta
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class CNN3D(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/cnn_3d'
        chk_n_mkdir(self.output_directory)
        self.model = self.build_model(input_shape, n_classes)
        if verbose:
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + '/model_init.hdf5')

    def build_model(self, input_shape, n_classes):
        ## input layer
        input_layer = Input(input_shape)

        ## convolutional layers
        conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 2), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 2), activation='relu')(conv_layer1)

        ## add max pooling to obtain the most imformatic features
        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

        conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 1), activation='relu')(pooling_layer1)
        conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 1), activation='relu')(conv_layer3)
        pooling_layer2 = MaxPool3D(pool_size=(2, 2, 1))(conv_layer4)

        ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pooling_layer2 = BatchNormalization()(pooling_layer2)
        flatten_layer = Flatten()(pooling_layer2)

        ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
        ## add dropouts to avoid overfitting / perform regularization
        dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=n_classes, activation='softmax')(dense_layer2)

        ## define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)

        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['acc'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)
        
        self.callbacks = [model_checkpoint, tb_cb]
        return model

