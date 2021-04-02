import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import pdb;
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
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
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))

    y = np.array(data["labels"])
    lbe = LabelEncoder()
    encoded_y = lbe.fit_transform(y)

    return X,encoded_y, X[0]


def prepare_datasets(test_size):

    # laod data
    X, y, positiveCase = load_data(DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Add pozitive case to test
    X_test[0] = positiveCase
    y_test[0] = 1

    # # create train/validation split
    # X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # # 3d array -> (130, 13, 1)
    # X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    # X_validation = X_validation[...,np.newaxis]
    # X_test = X_test[...,np.newaxis]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = prepare_datasets(0.3)
    eval_s = [(X_train, y_train),(X_test,y_test)]

    xgb_model = XGBClassifier().fit(X_train, y_train, eval_set=eval_s)
    y_pred = xgb_model.predict(X_test)
    print('Accuracy:', accuracy_score(y_pred, y_test))
    print('Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    y_pred_train = xgb_model.predict_proba(X_train)
    y_pred_train = y_pred_train[:, 0:2]
    y_pred_train_index = np.array([])
    for pred in y_pred_train:
        predicted_index = np.argmax(pred, axis=0)
        y_pred_train_index = np.append(y_pred_train_index, predicted_index)

    auc_score_train = roc_auc_score(y_train,y_pred_train_index)
    print('AUC Score Train: ', auc_score_train)

    confusion_matrix_results = confusion_matrix(y_train, y_pred_train_index)
    print('Train Confusion Matrix:')
    print(confusion_matrix_results)

    report_train = classification_report(y_train, y_pred_train_index, target_names=["covid", "not covid"])
    print('Report:', report_train)



    y_pred_test=xgb_model.predict_proba(X_test)
    y_pred_test = y_pred_test[:, 0:2]
    y_pred_test_index = np.array([])
    for pred in y_pred_test:
        predicted_index = np.argmax(pred, axis=0)
        y_pred_test_index = np.append(y_pred_test_index, predicted_index)
    
    auc_score_test=roc_auc_score(y_test,y_pred_test_index)
    print('AUC Score Test: ', auc_score_test)

    confusion_matrix_results = confusion_matrix(y_test, y_pred_test_index)
    print('Test Confusion Matrix:')
    print(confusion_matrix_results)

    report_test = classification_report(y_test, y_pred_test_index, target_names=["covid", "not covid"])
    print('Report:', report_test)

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train_index)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test_index)

    plt.plot(fpr_train, tpr_train, linestyle='--', color='orange', label='Train')
    plt.plot(fpr_test, tpr_test, linestyle='--', color='red', label='Test')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    train_precision, train_recall, _ = precision_recall_curve(y_train, y_pred_train_index)
    test_precision, test_recall, _ = precision_recall_curve(y_test, y_pred_test_index)

    plt.plot(train_recall, train_precision, marker='.', label='Train')
    plt.plot(test_recall, test_precision, marker='.', label='Test')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()



    # #Tuning
    # xgb_grid = {
    #     "n_estimators" : [50,100,500,1000],
    #     "subsample" : [0.2,0.4,0.6,0.8,1.0],
    #     "max_depth" : [3,4,5,6,7,8],
    #     "learning_rate" : [0.1, 0.01, 0.001, 0.0001],
    #     "min_samples_split" : [2,5,10,12]
    # }

    # from sklearn.model_selection import GridSearchCV

    # xgb_cv_model = GridSearchCV(xgb_model, xgb_grid, cv=5, n_jobs=-1, verbose=2)
    # xgb_cv_model.fit(X_train,y_train)
    # print(xgb_cv_model.best_params_)


    # xgb_tuned = XGBClassifier(learning_rate=0.1, max_dept=3, min_samples_split=2, n_estimators=500, subsample=0.4).fit(X_train,y_train)
    # y_tuned_pred = xgb_tuned.predict(X_test)
    # print('Tuned Accuracy:', accuracy_score(y_test, y_tuned_pred))
    # print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))