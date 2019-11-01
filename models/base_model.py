import time
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report, confusion_matrix
from abc import ABC, abstractmethod
from utils_basic import save_logs


class BaseModel(ABC):
    @abstractmethod
    def build_model(self, input_shape, n_classes):
        pass

    def fit(self, training_generator, testing_generator, samples_per_train_epoch, samples_per_test_epoch, epochs):

        start_time = time.time()
        hist = self.model.fit_generator(generator=training_generator,
                                        validation_data=testing_generator,
                                        samples_per_epoch=samples_per_train_epoch,
                                        validation_steps=samples_per_test_epoch,
                                        epochs=epochs,
                                        callbacks=self.callbacks,
                                        use_multiprocessing=False,
                                        workers=1,
                                        verbose=self.verbose)
        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + '/best_model.hdf5')
        test_loss, test_acc = model.evaluate_generator(testing_generator, steps=samples_per_test_epoch)
        print('test_loss:', test_loss, 'test_acc:', test_acc)

        y_pred = model.predict_generator(testing_generator, steps=samples_per_test_epoch)
        y_true = testing_generator.gen_class_labels()
        save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()

    def predict(self, testing_generator, samples_per_test_epoch):
        model = keras.models.load_model(self.output_directory + '/best_model.hdf5')
        test_loss, test_acc = model.evaluate_generator(testing_generator, steps=samples_per_test_epoch)
        print('test_loss:', test_loss, 'test_acc:', test_acc)
        y_pred = model.predict_generator(testing_generator, steps=samples_per_test_epoch)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = testing_generator.gen_class_labels()
        print('Confusion Matrix')

        target_names = testing_generator.integer_mapping_dict.keys()
        mat = confusion_matrix(y_true, y_pred)
        print(mat)
        df_cm = pd.DataFrame(mat, index=target_names, columns=target_names)
        plt.figure()
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        plt.show()

        print('Classification Report')
        print(classification_report(y_true, y_pred, target_names=target_names))

        keras.backend.clear_session()


