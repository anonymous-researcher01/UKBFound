import numpy as np
from survutil import (
    Survivaldata,
    ModelSaving,
    Cox,
    DeepSurv,
    EmbeddingModel,
    NegativeLogLikelihood,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython import display as IPD
import torch
import time
import os
import argparse
from survutil import Cox, EmbeddingModel, DeepSurv, c_index
from fileloader import load, loadindex


def modelchar(x):
    if x >= 0 and x <= 9:
        return str(x)
    elif x >= 10:
        return chr(65 + x - 10)


def train(net, n_epochs, waiting, train_loader, val_loader, device, lr):
    net.to(device)
    learning_rate = lr
    weight_decay = 0
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = NegativeLogLikelihood().to(device)
    early_break = ModelSaving(waiting=waiting, printing=True)
    train = []
    val = []
    val_lowest = np.inf
    for epoch in range(n_epochs):
        net.train()
        losses = []
        for batch_idx, (train_inputs, train_dates, train_labels) in enumerate(
            train_loader
        ):
            train_inputs, train_dates, train_labels = (
                train_inputs.to(device),
                train_dates.to(device),
                train_labels.to(device),
            )
            train_inputs = train_inputs.requires_grad_()
            train_outputs = net(train_inputs)
            loss = criterion(train_outputs, train_dates, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
        net.eval()
        val_losses = []
        for batch_idx, (val_inputs, val_dates, val_labels) in enumerate(val_loader):
            val_inputs, val_dates, val_labels = (
                val_inputs.to(device),
                val_dates.to(device),
                val_labels.to(device),
            )
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_dates, val_labels)
            val_losses.append(val_loss.data.mean().item())
        train.append(losses)
        val.append(val_losses)
        if np.mean(val_losses) < val_lowest:
            val_lowest = np.mean(val_losses)
            bestmodel = net
        early_break(np.mean(val_losses), net)

        train_L = [np.mean(x) for x in train]
        val_L = [np.mean(x) for x in val]

        if early_break.save:
            print(
                "Maximum waiting reached. Break the training.\t BestVal:{:.8f}\r".format(
                    min(val_L)
                )
            )
            break
        print(
            "Epoch: {}\tLoss: {:.8f}({:.8f})\t BestVal:{:.8f}\r".format(
                epoch + 1, train_L[-1], val_L[-1], min(val_L)
            ),
            end="",
            flush=True,
        )
        """try:
            plt.plot(list(range(1,len(train_L)+1)), np.log(train_L),'-o',label='Train',color="blue")
            plt.plot(list(range(1,len(val_L)+1)), np.log(val_L),'-x',label='Validation',color="red")
            IPD.display(plt.gcf())
            IPD.clear_output(wait=True)
            time.sleep(0.01)
        except KeyboardInterrupt:
            break"""
    print(f'Exit at {epoch}')
    return bestmodel


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="parse")
    parser.add_argument("category", type=int)
    parser.add_argument("model", type=int)
    parser.add_argument("hyperp", type=int)
    parser.add_argument("Xtype", type=int)
    parser.add_argument("gpu", type=int)
    parser.add_argument("string", type=str)
    args = parser.parse_args()
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    category = args.category
    model = args.model
    hyperp = args.hyperp
    image_X = args.Xtype
    loc = args.string
    device = torch.device("cuda")
    imageX = image_X
    only = False
    waiting = 15
    epoch = 1000
    learning_rate = 0.005
    X, Y, E = load(image_X=imageX, category=category, only=only)
    index = np.isnan(E)
    E = np.where(index, torch.tensor(0), E)
    numbers = list(range(X.shape[0]))
    *_, trainindex, valindex, testindex = loadindex(imageX)
    trainset = Survivaldata(X, Y, E, trainindex)
    valset = Survivaldata(X, Y, E, valindex)
    testset = Survivaldata(X, Y, E, valindex)
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])
    if model == 0:  #'cox':
        nnet = Cox(shape_data, shape_label)
    elif model == 1:  #'deepsurv':
        nnet = DeepSurv(shape_data, shape_label, 5, 300)
    elif model == 2:  #'popdx':
        label_emb = np.load(
            "../../data/Embedding/phe.npy", allow_pickle=True
        )
        hidden_size = 400
        nnet = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)
    elif model == 3:  #'mith':
        label_emb = np.load(
            "../../data/Embedding/conv.npy", allow_pickle=True
        )
        hidden_size = 400
        nnet = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)
    nnet = train(nnet, epoch, waiting, train_loader, val_loader, device, learning_rate)
    nnet.to("cuda")
    whole_loader = DataLoader(testset, batch_size=len(testset))
    xtest, ytest, etest = next(iter(whole_loader))
    ytest.cpu()
    etest.cpu()
    out = nnet(xtest.to("cuda")).cpu().detach().numpy()
    cindex = []
    for j in range(out.shape[1]):
        try:
            na_indices = np.where(np.isnan(ytest[:, j]))[0]
            o = np.delete(-out[:, j], na_indices)
            y = np.delete(ytest[:, j], na_indices)
            e = np.delete(etest[:, j], na_indices)
            cindex.append(c_index(o, y, e))
        except Exception as e:
            cindex.append(np.nan)
    loc='../../results/Disease_diagnosis/'
    try:
        os.mkdir(f"./{loc}surv")
    except:
        pass
    np.save(f"./{loc}surv/{category}{modelchar(model)}{imageX}_{hyperp}", cindex)
    # np.save(f"./surv/{category}{modelchar(model)}{imageX}out", out)
    # np.save(f"./surv/{category}{modelchar(model)}{imageX}lab", [ytest,etest])
    np.save(f"./{loc}surv/{imageX}lab", [ytest, etest])
    torch.save(nnet, f"./{loc}surv/{category}{modelchar(model)}{imageX}_{hyperp}model")
    print("complete")
