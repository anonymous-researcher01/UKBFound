import pickle
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import time
import matplotlib.pyplot as plt
from config import *
def renderresult(label, predict, filename="", supress=True):
    na_indices = np.where(np.isnan(label) | np.isnan(predict))[0]
    predict = np.delete(predict, na_indices)
    label = np.delete(label, na_indices)
    fpr, tpr, thresholds = metrics.roc_curve(label, predict, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    if supress:
        return roc_auc
    plt.figure(dpi=500)
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    try:
        plt.show()
    except:
        pass
    return roc_auc


def extract_importance(model, count=30):
    inv = 1
    absolute = True
    try:
        importance = model.feature_importance()
    except:
        try:
            importance = model.get_feature_importance()
        except:
            try:
                importance = model.feature_importances_
            except:
                try:
                    importance = np.array(
                        list(model.get_score(importance_type="gain").values())
                    )
                except:
                    try:
                        importance = np.log(model.pvalues)
                        coef = model.params
                        absolute = False
                        inv = -1
                    except:
                        try:
                            importance = model.coef_
                        except:
                            pass
    if absolute:
        importance = abs(importance)
    indices = np.argsort(inv * importance)
    top = indices[-1 * count :]
    # top=indices
    return top, importance


os.environ["CUDA_VISIBLE_DEVICES"] = str(7)
from datetime import datetime
from fileloader import load,loadindex
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier

np.random.seed(0)
path = f"../../results/Disease_diagnosis/{folder}rf/"
try:
    os.mkdir(path)
except:
    pass
for image_X in imglist:
    for category in catlist:
        Xdata, _,lab = load(image_X, category)
        print(Xdata.shape,lab.shape)
        numbers = list(range(lab.shape[0]))
        trainindex, valindex, _, _ = train_test_split(numbers, numbers, test_size=0.2)
        testindex, valindex, _, _ = train_test_split(valindex, valindex, test_size=0.5)
        testindex = np.array(testindex)
        trainindex = np.array(trainindex)
        valindex = np.array(valindex)
        *_, trainindex, valindex, testindex = loadindex(image_X)
        result = {}
        
        for i in range(lab.shape[1]):
            if image_X == 0:
                addstr = ""
            else:
                addstr = "_5w"
            if str(i) + "_" + str(category) + addstr not in os.listdir(path):
                print((str(i), str(category), str(image_X)))
                y_col = lab[:, i]
                val_mask = ~np.isnan(y_col[valindex])
                test_mask = ~np.isnan(y_col[testindex])
                train_mask = ~np.isnan(y_col[trainindex])
                filtered_trainindex = trainindex[train_mask]
                filtered_valindex = valindex[val_mask]
                filtered_testindex = testindex[test_mask]
                y_train = y_col[filtered_trainindex]
                X_train = Xdata[filtered_trainindex]
                y_val = y_col[filtered_valindex]
                X_val = Xdata[filtered_valindex]
                y_test = y_col[filtered_testindex]
                X_test = Xdata[filtered_testindex]
                if image_X == 0:
                    addstr = ""
                else:
                    addstr = "_5w"
                try:
                    '''model_rf = RandomForestClassifier(
                        n_estimators=100,
                        n_jobs=100,
                    )
                    model = model_rf
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    importance = model_rf.feature_importances_'''
                    model_rf = RandomForestClassifier(
                        n_estimators=100,
                    )
                    model_rf.fit(np.array(X_train,dtype=np.float32), np.array(y_train,dtype=np.float32))
                    y_pred = model_rf.predict_proba(X_test)[:, 1]
                    train_pred = model_rf.predict_proba(X_train)[:, 1]
                    res = renderresult(y_test, y_pred, supress=True)
                    train_res = renderresult(y_train, train_pred, supress=True)
                    print(res)
                    result[i] = [i, 'importance', res, train_res]
                    with open(
                        path + str(i) + "_" + str(category) + addstr, "wb+"
                    ) as file:
                        pickle.dump(result[i], file)
                        file.close()
                except Exception as e:
                    with open(
                        path + str(i) + "_" + str(category) + addstr, "w+"
                    ) as file:
                        print(e)
                        file.write(str(e))
                        file.close()
