import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import pdb;
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt


DATA_PATH="data.json"

def load_data(data_path):
    """Loads training dataset from json file

    :param data_path(str): path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    mfccsArray = []
    for i, mfcc in enumerate(data['MFCCs']):
        if np.array(mfcc).shape == (431, 13):
            mfccsArray.append(mfcc)
        elif np.array(mfcc).shape[0] < 431:
            zeros = np.zeros([431,13])
            zeros[:np.array(mfcc).shape[0], :np.array(mfcc).shape[1]] = np.array(mfcc)
            mfccsArray.append(zeros)
        elif np.array(mfcc).shape[0] > 431:
            mfccArray = np.array(mfcc)
            mfccsArray.append(mfccArray[:431, :13])
        

    X = np.array(mfccsArray)

    y = np.array(data["labels"])

    lbe = LabelEncoder()
    encoded_y = lbe.fit_transform(y)

    return X,encoded_y, X[0]


def prepare_datasets(test_size, validation_size):

    # laod data
    X, y, positiveCase = load_data(DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Add pozitive case to test
    X_test[0] = positiveCase
    y_test[0] = 1

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]


    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create mdoel
    model = keras.Sequential()


    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
        # max pooling
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))


    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
        # max pooling
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))


    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu", input_shape=input_shape))
        # max pooling
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))


    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.1))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X,y):

    X = X[np.newaxis, ...]

    y_pred = model.predict(X) # X -> 3d array (?1?, 130, 13, 1)

    #y_pred 2d array -> [[0.1,0.2,..., 0.3]]
    # extract index with max value
    predicted_index = np.argmax(y_pred, axis=1) # [3]

    print("Expected index: {}, Predicted index: {} ". format(y, predicted_index))

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
def statistics(type, pred, real):

    auc_score=roc_auc_score(real,pred)
    if type == 'test':
        print('Test AUC:', auc_score)
    elif type == 'train':
        print('Train AUC:', auc_score)

    confusion_matrix_results = confusion_matrix(real, pred)
    if type == 'test':
        print('Test Confusion Matrix:', confusion_matrix_results)
    elif type == 'train':
        print('Train Confusion Matrix:', confusion_matrix_results)

    report = classification_report(real, pred, target_names=["covid", "not covid"])
    if type == 'test':
        print('Test Report:', report)
    elif type == 'train':
        print('Train Report:', report)

if __name__ == "__main__":
    
    # create train, validation and test sets # test_size, validation_size
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.2, 0.2)

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=5)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Accuracy Test {}'.format(test_accuracy))
    print('Error Test {}'.format(test_error))
    
    # make prediction on a sample
    X = X_test[2]
    y = y_test[2]

    #predict(model, X,y)
    #plot the accuracy and error over the epochs
    plot_history(history)
    
    y_pred_train = model.predict_proba(X_train)
    y_pred_train_index = np.array([])
    for pred in y_pred_train:
        predicted_index = np.argmax(pred, axis=0)
        y_pred_train_index = np.append(y_pred_train_index, predicted_index)

    auc_score_train=roc_auc_score(X_train,y_pred_train_index)
    print('AUC Score Train: ', auc_score_train)

    y_pred_validation = model.predict_proba(X_validation)
    y_pred_validation_index = np.array([])
    for pred in y_pred_validation:
        predicted_index = np.argmax(pred, axis=0)
        y_pred_validation_index = np.append(y_pred_validation_index, predicted_index)

    auc_score_validation=roc_auc_score(X_validation,y_pred_validation_index)
    print('AUC Score Validation: ', auc_score_validation)

    y_pred_test=model.predict_proba(X_test)
    y_pred_test_index = np.array([])
    for pred in y_pred_test:
        predicted_index = np.argmax(pred, axis=0)
        y_pred_test_index = np.append(y_pred_test_index, predicted_index)
    
    auc_score_test=roc_auc_score(X_test,y_pred_test_index)
    print('AUC Score Test: ', auc_score_test)
         