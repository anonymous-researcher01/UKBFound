import pandas as pd
import numpy as np
import os
from config import *


def loadindex(image_X, loc="../../results/Preprocess/"):
    if image_X == 0:
        trainindex = pd.read_csv(f"{loc}train_index.csv")["train"].to_numpy()
        valindex = pd.read_csv(f"{loc}val_index.csv")["val"].to_numpy()
        testindex = pd.read_csv(f"{loc}test_index.csv")["test"].to_numpy()
        return trainindex, valindex, testindex
    elif image_X == 1:
        trainindex = pd.read_csv(f"{loc}image_train_index.csv")["train"].to_numpy()
        valindex = pd.read_csv(f"{loc}image_val_index.csv")["val"].to_numpy()
        testindex = pd.read_csv(f"{loc}image_test_index.csv")["test"].to_numpy()
        imageindex = pd.read_csv(f"{loc}image_index.csv")["image"].to_numpy()
        return imageindex, trainindex, valindex, testindex


def load(
    image_X,
    category,
    only=False,
    xloc=Xblocklocation,
    yloc="../../results/cache",
    MRIloc=None,
    fullimgspec=0,
):  
    if category ==6:
        y = np.load(f"{yloc}/coximg.npy")
        e = np.load(f"{yloc}/0-1img.npy")
    else:
        y = np.load(f"{yloc}/cox.npy")
        e = np.load(f"{yloc}/0-1.npy")
        
    if category < 6:
        Xdata = np.load(f"{xloc}blk{category}.npy")
    else:
        Xdata = np.load(f"{xloc}blk5.npy")
    if image_X == 1:
        raise ValueError("Not implemented")
    elif image_X == 0:
        if category == 6:
            mri = pd.read_csv('../../results/Process_missingness/mripc_imputed.csv').to_numpy()
            Xdata = np.hstack((Xdata, mri))
    return Xdata, y, e
