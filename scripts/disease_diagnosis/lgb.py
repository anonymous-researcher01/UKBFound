import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
import matplotlib.pyplot as plt

from datetime import datetime
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
import lightgbm as lgb
from fileloader import load,loadindex
from config import *
np.random.seed(0)
torch.manual_seed(0)

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


for image_X in imglist:
    for category in catlist:
        Xdata, _,lab = load(image_X, category)
        
        print(Xdata.shape,lab.shape)
        numbers = list(range(lab.shape[0]))
        *_, trainindex, valindex, testindex = loadindex(image_X)
        result = {}
        path = f"../../results/Disease_diagnosis/{folder}lgb/"
        try:
            os.mkdir(path)
        except:
            pass
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
                train_data = lgb.Dataset(X_train, label=y_train)
                validation_data = lgb.Dataset(X_val, label=y_val)
                test_data = lgb.Dataset(X_test, label=y_test)
                num_round = 200
                param = {
                    "objective": "binary",
                    "metric": "auc",
                    "nthread": 100,
                }
                lgb_model = lgb.train(
                    param,
                    train_data,
                    num_round,
                    valid_sets=[validation_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=50)]
                )
                
                y_pred = lgb_model.predict(
                    X_test, num_iteration=lgb_model.best_iteration
                )
                train_pred=lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
                importance = lgb_model.feature_importance()

                try:
                    res = renderresult(y_test, y_pred, supress=True)
                    train_res = renderresult(y_train, train_pred, supress=True)
                    print(res)
                    result[i] = [i, importance, res, train_res]
                    with open(
                        path + str(i) + "_" + str(category) + addstr, "wb+"
                    ) as file:
                        pickle.dump(result[i], file)
                        file.close()
                except Exception as e:
                    with open(
                        path + str(i) + "_" + str(category) + addstr, "wb+"
                    ) as file:
                        file.write(e)
                        file.close()
