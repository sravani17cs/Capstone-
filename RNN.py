from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

# Helper libraries
import numpy as np
import pickle
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split as split
import matplotlib.pyplot as plt

# Customizable Values
LSTM_OUTPUT_SIZE = 64
DENSE_LAYER1_OUTPUT_SIZE = 32
DENSE_LAYER2_OUTPUT_SIZE = 34 #Number of classes
TEST_DATA_SIZE = 0.1
GAMMA = 0
ALPHA = 1.0

X = []
y = []
encoding = []
labels = []
int_to_char = []
char_to_int = []

fl = tfa.losses.SigmoidFocalCrossEntropy(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
fl.gamma=GAMMA
fl.alpha=ALPHA

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def main():

    global X
    global y
    global labels

    y = keras.utils.to_categorical(y, num_classes=len(np.unique(y)))
    X_train, X_test, y_train, y_test = split(X, y, test_size=TEST_DATA_SIZE, shuffle=True, stratify=y)

    X_train = np.array(X_train)
    X_test = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = keras.Sequential([
        tf.keras.layers.LSTM(LSTM_OUTPUT_SIZE, input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(DENSE_LAYER1_OUTPUT_SIZE),
        tf.keras.layers.Dense(DENSE_LAYER2_OUTPUT_SIZE, activation='softmax')])
    model.summary()



    model.compile(optimizer='nadam',
                  loss=fl,
                  metrics=METRICS)

    print(model.layers[0].output_shape)
    print(model.layers[1].output_shape)
    print(model.layers[2].output_shape)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(X_train, y_train, epochs=350, verbose=1, batch_size=128, validation_split=0.1, callbacks=[tensorboard_callback])

    print("\nHistory dict: ", history.history)

    result = model.predict(X_test)
    y_true_int = [[y.argmax()] for y in y_test]
    y_pred_int = [[y.argmax()] for y in result]
    y_true = []
    y_pred = []
    for i in y_true_int:
        y_true.append([int_to_char[i[0]]])
    for i in y_pred_int:
        y_pred.append([int_to_char[i[0]]])
    total = y_pred + y_true
    labels = np.unique(total)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(classification_report(y_true, y_pred, labels=labels))
    plot_conf_matrix(cm, y_true, y_pred, labels, normalize=True)
    plot_conf_matrix(cm, y_test, y_pred, labels, normalize=False)

    return



# from scikit-learn https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_conf_matrix(cm, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def getData():
    global X
    global y
    global encoding
    global int_to_char
    global char_to_int
    global labels
    with open("data/filteredFeatures", 'rb') as file:
        X = pickle.load(file)
    with open("data/filteredAnnotations", 'rb') as file:
        y = pickle.load(file)

    print("OnesHot Encoding annotations")
    temp = ""
    new_y = []
    for y_row in y:
        for i in np.unique(y_row):
            if str(i) not in temp:
                temp = temp + str(i)
    encoding = temp
    labels = encoding
    char_to_int = dict((c, i) for i, c in enumerate(encoding))
    int_to_char = dict((i, c) for i, c in enumerate(encoding))
    for y_row in y:
        integer_encoding = [char_to_int[char] for char in y_row]
        onehot_encoding = list()
        for value in integer_encoding:
            symbol = [0 for _ in range(len(encoding))]
            symbol[value] = 1
            onehot_encoding.append(symbol)
        # new_y.append(onehot_encoding)
        new_y.append(integer_encoding[0])
    y = new_y


if __name__ == '__main__':
    start_time = time.time()
    getData()
    local_dir = os.path.dirname(os.path.abspath(__file__))
    main()
    print("Process took ", time.time()-start_time)
