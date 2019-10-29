from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class VCNN1(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/vcnn1_3d'
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
        conv_layer1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
        pooling_layer1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_layer1)

        conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(pooling_layer1)

        conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_layer2)
        pooling_layer2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_layer3)
        dropout_layer =Dropout(0.5)(pooling_layer2)

        dense_layer = Dense(units=2048, activation='relu')(dropout_layer)
        output_layer = Dense(units=n_classes, activation='softmax')(dense_layer)

        ## define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)

        model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)
        
        self.callbacks = [model_checkpoint, tb_cb]
        return model

