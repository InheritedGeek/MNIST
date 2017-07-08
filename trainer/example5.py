import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
# import tensorflow as tf
# from tensorflow.python.lib.io import file_io
import pickle
from datetime import datetime
import time
import argparse

batch_size = 10000
num_classes = 10
epochs = 2
logs_path = './tmp/example-5/' + datetime.now().isoformat()

def train_model(train_file='train.csv', **args):
    # Here put all the main training code in this function
    # file = file_io.FileIO(train_file, mode='r')

    file = pd.read_csv("train.csv").values
    # test = pd.read_csv("test.csv").values

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # Reshaping feature data for 2d Conv. layer
    x_train = file[:, 1:].reshape(file.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype(float)
    x_train /= 255.0

    # Converting output data
    y_train = keras.utils.to_categorical(file[:, 0], num_classes)

    # Splitting into training & validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    # Building Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
              optimizer="Adadelta",
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))

    score = model.evaluate(x_val, y_val, verbose=0)

    print 'Test loss:', score[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')

    train_model(**arguments)

# x_test = test.reshape(test.shape[0], 28, 28, 1)
# x_test = x_test.astype(float)
# x_test /= 255.0

# predictions = model.predict_classes(x_test)
#
# submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": predictions})
#
# submissions.to_csv("Prediction.csv", index=False, header=True)
