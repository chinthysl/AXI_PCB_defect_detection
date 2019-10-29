from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Maximum
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Adadelta
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks

from utils_datagen import TrainValTensorBoard
from utils_basic import chk_n_mkdir
from models.base_model import BaseModel


class MVCNN_CNN(BaseModel):
    def __init__(self, output_directory, input_shape, n_classes, verbose=False):
        self.output_directory = output_directory + '/mvcnn_cnn'
        chk_n_mkdir(self.output_directory)
        self.model = self.build_model(input_shape, n_classes)
        if verbose:
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + '/model_init.hdf5')

    def cnn(self, input_shape):
        # ## input layer
        input_layer = Input(input_shape)

        ## convolutional layers
        conv_layer1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(input_layer)
        conv_layer2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(conv_layer1)

        ## add max pooling to obtain the most imformatic features
        pooling_layer1 = MaxPooling2D(pool_size=(2, 2))(conv_layer2)

        conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(pooling_layer1)
        conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer3)
        pooling_layer2 = MaxPooling2D(pool_size=(2, 2))(conv_layer4)

        ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pooling_layer2 = BatchNormalization()(pooling_layer2)
        flatten_layer = Flatten()(pooling_layer2)

        # input layer
        # input_layer = Input(input_shape)
        # conv_layer1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
        # conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_layer1)
        # pooling_layer1 = MaxPooling2D(pool_size=(4, 4))(conv_layer2)
        # # dropout_layer1 = Dropout(0.25)(pooling_layer1)
        # dropout_layer1 =BatchNormalization()(pooling_layer1)
        # flatten_layer = Flatten()(dropout_layer1)

        # Create model.
        model = Model(input_layer, flatten_layer)
        return model

    def build_model(self, input_shape, n_classes):
        cnn_1 = self.cnn(input_shape)
        input_1 = cnn_1.input
        output_1 = cnn_1.output
        cnn_2 = self.cnn(input_shape)
        input_2 = cnn_2.input
        output_2 = cnn_2.output
        cnn_3 = self.cnn(input_shape)
        input_3 = cnn_3.input
        output_3 = cnn_3.output
        cnn_4 = self.cnn(input_shape)
        input_4 = cnn_4.input
        output_4 = cnn_4.output

        concat_layer = Maximum()([output_1, output_2, output_3, output_4])
        concat_layer = Dropout(0.5)(concat_layer)
        dense_layer1 = Dense(units=1024, activation='relu')(concat_layer)
        dense_layer1 = Dropout(0.5)(dense_layer1)
        output_layer = Dense(n_classes, activation='softmax', name='predictions')(dense_layer1)

        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=[output_layer])
        model.summary()
        plot_model(model, to_file=self.output_directory + '/model_graph.png', show_shapes=True, show_layer_names=True)
        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.01), metrics=['accuracy'])

        # model save
        file_path = self.output_directory + '/best_model.hdf5'
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Tensorboard log
        log_dir = self.output_directory + '/tf_logs'
        chk_n_mkdir(log_dir)
        tb_cb = TrainValTensorBoard(log_dir=log_dir)
        
        self.callbacks = [model_checkpoint, tb_cb]
        return model

