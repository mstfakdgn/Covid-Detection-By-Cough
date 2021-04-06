import json
import numpy as np
from sklearn.model_selection import train_test_split, KFold
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

    return X_train, X_test, y_train, y_test

def statistics(type, pred, real, index):
    accuracy = accuracy_score(real, pred)
    error = np.sqrt(mean_squared_error(real, pred))

    print("Accuracy--Index => \n" + str(index + 1) , type, accuracy)
    Raports['accuricies'].append([accuracy.tolist(), str(index + 1), str(type)])
    print("Error--Index => \n" + str(index + 1), type, error)
    Raports['errors'].append([error.tolist(), str(index + 1), str(type)])
    print('------------------------------------------------------------------')

    try:
        auc_score = roc_auc_score(real,pred)
        print("AUC Score--Index => \n" + str(index + 1), type, auc_score)
        Raports['aucScores'].append([auc_score.tolist(), str(index + 1), str(type)])
    except ValueError:
        pass

    print('------------------------------------------------------------------')
    confusion_matrix_results = confusion_matrix(real, pred)
    print("Train Confusion Matrix--Index => \n"+ str(index + 1), type)
    Raports['counfusionMatrix'].append([confusion_matrix_results.tolist(), str(index + 1), str(type)])
    print(confusion_matrix_results, str(index + 1), str(type))

    print('------------------------------------------------------------------')
    report = classification_report(real, pred, target_names=["covid", "not covid"])
    print("Report--Index => \n" + str(index + 1), type, report)
    Raports['raports'].append([report, str(index + 1), str(type)])
    print('------------------------------------------------------------------')

if __name__ == "__main__":
    
    Raports = {
        "raports": [],
        "counfusionMatrix": [],
        "aucScores": [],
        "accuricies": [],
        "errors": []
    }

    rocsTrain = []
    rocsTest = []
    precisionsTrain = []
    precisionsTest = []

    X, y, positiveCase = load_data(DATA_PATH)


    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        eval_s = [(X_train, y_train),(X_test,y_test)]

        for j,case in enumerate(y):
            if case == 1:
                y_test[0] = 1
                X_test[0] = X[j]
                y_train[0] = 1
                X_train[0] = X[j]
            elif case == 0:
                y_test[1] = 0
                X_test[1] = X[j]
                y_train[1] = 0
                X_train[1] = X[j]

        xgb_model = XGBClassifier().fit(X_train, y_train, eval_set=eval_s)
        

        y_pred_train = xgb_model.predict_proba(X_train)
        y_pred_train = y_pred_train[:, 0:2]
        y_pred_train_index = np.array([])

        for pred in y_pred_train:
            predicted_index = np.argmax(pred, axis=0)
            y_pred_train_index = np.append(y_pred_train_index, predicted_index)

        statistics('Train', y_train,y_pred_train_index, i)


        y_pred_test=xgb_model.predict_proba(X_test)
        y_pred_test = y_pred_test[:, 0:2]
        y_pred_test_index = np.array([])
        for pred in y_pred_test:
            predicted_index = np.argmax(pred, axis=0)
            y_pred_test_index = np.append(y_pred_test_index, predicted_index)
        
        statistics('Test', y_test,y_pred_test_index, i)
        
    
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train_index)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test_index)

        rocsTrain.append([fpr_train, tpr_train])
        rocsTest.append([fpr_test, tpr_test])



        train_precision, train_recall, _ = precision_recall_curve(y_train, y_pred_train_index)
        test_precision, test_recall, _ = precision_recall_curve(y_test, y_pred_test_index)

        precisionsTrain.append([train_precision, train_recall])
        precisionsTest.append([test_precision, test_recall])

    fig, axs = plt.subplots(5)
    for j, roc in enumerate(rocsTrain):
        # create train
        axs[j].plot(roc[0], roc[1], linestyle='--', label='Train' + str(j+1))
        axs[j].plot(rocsTest[j][0], rocsTest[j][1], linestyle='--',label="Test" + str(j+1))
        axs[j].set_ylabel("Roc")
        axs[j].legend(loc="lower right")
        axs[j].set_title("Roc")
        
    plt.show()

    fig, axs = plt.subplots(5)
    for k, precisionRecall in enumerate(precisionsTrain):
        # create train
        axs[k].plot(precisionRecall[0], precisionRecall[1], linestyle='--', label='Train' + str(k+1))
        axs[k].plot(precisionsTest[k][0], precisionsTest[k][1], linestyle='--',label="Test" + str(k+1))
        axs[k].set_ylabel("Precision Recall")
        axs[k].legend(loc="lower right")
        axs[k].set_title("Precision Recall")
        
    plt.show()

    with open("./raportsXGBoost.json", "w") as fp:
        json.dump(Raports, fp, indent=4)
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