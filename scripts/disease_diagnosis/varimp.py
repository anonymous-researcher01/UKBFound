import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
    KernelShap,
    DeepLiftShap,
)
import os
import sys
from config import *
from fileloader import load,loadindex
if __name__ == "__main__":
    args = sys.argv[1:]
    st = int(args[0])
    end = int(args[1])
    gpu = int(args[2])
    folder = str(args[3])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    class pheNN(nn.Module):
        def __init__(self, input_size, output_size, depth, width):
            super(pheNN, self).__init__()
            layers = []
            for i in range(depth):
                layers.append(nn.Linear(width, width))
            self.inlayer = nn.Linear(input_size, width)
            self.layers = nn.ModuleList(layers)
            self.outlayer = nn.Linear(width, output_size)

        def forward(self, x):
            x = self.inlayer(x)
            for layer in self.layers:
                x = layer(x)
                x = nn.ReLU()(x)
            x = self.outlayer(x)
            return x

        def initialize(self):
            pass

    class ukbdata(Dataset):
        def __init__(self, dataframe, labels):
            self.df = dataframe
            self.label = labels

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            data = torch.from_numpy(self.df[idx]).float()
            label = torch.from_numpy(self.label[idx]).float()
            return data, label

    np.random.seed(0)
    torch.manual_seed(0)
    loc='../../results/Disease_diagnosis/'
    cat=6
    Xdata, _,lab = load(0,cat)
    print("loaded")
    numbers = list(range(lab.shape[0]))
    trainindex, valindex, _, _ = train_test_split(numbers, numbers, test_size=0.2)
    testindex, valindex, _, _ = train_test_split(valindex, valindex, test_size=0.5)
    *_, trainindex, valindex, testindex = loadindex(0)
    trainset = ukbdata(Xdata[trainindex], lab[trainindex])
    valset = ukbdata(Xdata[valindex], lab[valindex])
    testset = ukbdata(Xdata[testindex], lab[testindex])
    train_loader = DataLoader(trainset, batch_size=3000, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])
    whole_loader = DataLoader(testset, batch_size=3000, shuffle=True)
    device = "cuda"
    model = torch.load(f"./{loc}pred/{cat}10_3model").to(device)
    import pickle
    gs = GradientShap(model)
    dl = DeepLift(model)
    ig = IntegratedGradients(model)
    ks = KernelShap(model)
    dls = DeepLiftShap(model)
    train, labels = next(iter(train_loader))
    X_train = train.to(device)
    putdir = f"./{loc}imp/"
    try:
        os.mkdir(putdir)
    except:
        pass
    for tg in range(st, end):
        if str(tg) in os.listdir(putdir):
            continue
        print(tg)
        dumpdict = {"dl": [], "gs": [], "ig": []}
        for t in range(len(whole_loader)):
            inputs, labels = next(iter(whole_loader))

            X_test = inputs.to(device)
            #igout = ig.attribute(X_test, target=tg).detach().cpu().numpy()
            gsout = gs.attribute(X_test, X_train, target=tg).detach().cpu().numpy()
            #dlout = dl.attribute(X_test, target=tg).detach().cpu().numpy()
            #dumpdict["ig"].append(igout)
            dumpdict["gs"].append(gsout)
            #dumpdict["dl"].append(dlout)
            # dumpdict['ks'].append(ks.attribute(X_test,target=tg).detach().cpu().numpy())
            # dumpdict['dls'].append(dls.attribute(X_test,X_train,target=tg).detach().cpu().numpy())
        # with open(putdir + str(tg), "wb+") as file:
        #     pickle.dump(np.concatenate(dumpdict['gs']), file)
        with open(putdir + str(tg)+'sumed', "wb+") as file:
            pickle.dump(np.concatenate(dumpdict['gs']).sum(0), file)
