"""
module for student grade prediction
"""
import pandas as pd
import tensorflow as tf
import keras_tuner as kt  # pylint: disable=E0401
from sklearn.preprocessing import StandardScaler
from tensorflow import keras  # pylint: disable=E0401
from sklearn.model_selection import train_test_split  # pylint: disable=E0401

df = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
data_x = df.drop(columns=['Outcome'])
data_y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


class KerasClassifier:

    def __init__(self):
        pass

    def model_builder(self, hparameter):
        model = keras.Sequential()
        for i in range(hparameter.Int('num_layers', 2, 30)):
            model.add(keras.layers.Dense(units=hparameter.Int('units_' + str(i), min_value=8, max_value=512, step=16),
                                         input_shape=(8,),
                                         activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hparameter.Choice('learning_rate',
                                  values=[1e-2, 1e-3, 1e-4])),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        # model.fit(x_train, y_train, epochs=3)
        return model

    def run_tuner(self):
        tuner = kt.RandomSearch(
            self.model_builder,
            objective='val_accuracy',
            max_trials=5,
            project_name='EKU-2021'
        )
        tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
        best_model.save("model.h5")
