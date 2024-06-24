import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import warnings
import torch
import torch.nn as nn
import os
from fileloader import load,loadindex
import torch.nn.functional as F

warnings.filterwarnings("ignore")
from survutil import (
    Survivaldata,
    ModelSaving,
    Cox,
    DeepSurv,
    EmbeddingModel,
    NegativeLogLikelihood,
)

def renderresult(label, predict, supress=True):
    na_indices = np.where(np.isnan(label) | np.isnan(predict))[0]
    predict = np.delete(predict, na_indices)
    label = np.delete(label, na_indices)
    fpr, tpr, thresholds = metrics.roc_curve(label, predict, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    if supress:
        return roc_auc
    pyplot.figure()
    lw = 2
    pyplot.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver operating characteristic")
    pyplot.legend(loc="lower right")
    try:
        pyplot.show()
    except:
        pass
    return roc_auc


class BCEWithLogitsLossIgnoreNaN(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        mask = ~torch.isnan(target)
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)
        return F.binary_cross_entropy_with_logits(
            masked_input,
            masked_target,
        )


def custom_loss(pred, target):
    nans = torch.isnan(target)
    pred = torch.where(nans, torch.tensor(1), pred)
    target = torch.where(nans, torch.tensor(1), target)
    bceloss = torch.nn.BCEWithLogitsLoss()(pred, target)
    return bceloss


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


class ModelSaving:
    def __init__(self, waiting=3, printing=True):
        self.patience = waiting
        self.printing = printing
        self.count = 0
        self.best = None
        self.save = False

    def __call__(self, validation_loss):
        if not self.best:
            self.best = -validation_loss
        elif self.best <= -validation_loss:
            self.best = -validation_loss
            self.count = 0
        elif self.best > -validation_loss:
            self.count += 1
            print(f"Validation loss has increased: {self.count} / {self.patience}.")
            if self.count >= self.patience:
                self.save = True


def train(net, n_epochs=1000, waiting=5):
    net.initialize()
    net.cuda()

    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = custom_loss
    early_break = ModelSaving(waiting=waiting, printing=True)
    train = []
    val = []
    val_lowest = np.inf
    for epoch in range(n_epochs):
        #print("starting epoch " + str(epoch))
        net.train()
        losses = []
        for batch_idx, (train_inputs, train_labels) in enumerate(train_loader):
            train_inputs, train_labels = Variable(train_inputs.to(device)), Variable(
                train_labels.to(device)
            )
            train_inputs.requires_grad_()

            train_outputs = net(train_inputs)
            loss = criterion(train_outputs, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
        net.eval()
        val_losses = []
        for batch_idx, (val_inputs, val_labels) in enumerate(val_loader):
            val_inputs, val_labels = Variable(val_inputs.to(device)), Variable(
                val_labels.to(device)
            )
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_losses.append(val_loss.data.mean().item())

        train.append(losses)
        val.append(val_losses)
        if np.mean(val_losses) < val_lowest:
            val_lowest = np.mean(val_losses)
            bestmodel = net
        early_break(np.mean(val_losses))
        if early_break.save:
            print("Maximum waiting reached. Break the training.")
            break
    return bestmodel


def modelchar(x):
    if x >= 0 and x <= 9:
        return str(x)
    elif x >= 10:
        return chr(65 + x - 10)


class POPDxModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(feature_num, hidden_size, bias=True),
                nn.Linear(hidden_size, y_emb.shape[1], bias=True),
            ]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        x = torch.relu(x)
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class POPDxModelC(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModelC, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(feature_num, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, y_emb.shape[1], bias=True),
            ]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i <= 2:
                x = torch.relu(x)
        x = torch.matmul(x, torch.transpose(torch.tensor(self.y_emb), 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class POPDxModelC1(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModelC1, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(feature_num, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, y_emb.shape[1], bias=True),
            ]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


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


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        return out

    def initialize(self):
        pass


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
    #Xdata, _,lab = load(image_X, category,fullimgspec=imgimp)
    Xdata, _,lab = load(image_X, category,fullimgspec=0)

    print(
        "model", model, "cat", category, len(Xdata), "params", hyperp, "Xtype", image_X
    )
    print(Xdata.shape, lab.shape)
    print("loaded")
    numbers = list(range(lab.shape[0]))
    *_, trainindex, valindex, testindex = loadindex(image_X)
    learning_rate = 0.0001
    weight_decay = 0
    device = torch.device("cuda")
    trainset = ukbdata(Xdata[trainindex], lab[trainindex])
    valset = ukbdata(Xdata[valindex], lab[valindex])
    testset = ukbdata(Xdata[testindex], lab[testindex])
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])
    if model == 0:
        label_emb = np.load(f"../../data/Embedding/phe.npy", allow_pickle=True)
        label_emb=torch.tensor(label_emb,device=device).float()
        parampair=[25,50,100,200,400,500,1000]
        hidden_size=parampair[hyperp]
        #net = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)
        net = POPDxModelC1(shape_data, shape_label, hidden_size, label_emb)
    elif model == 1:
        parampair=[(1,25),(1,50),(2,50),(2,100),(3,50),(3,100),(3,200),(5,500),(6,300),(6,400),(6,500),(7,500),(8,500),(10,1000)]
        param=parampair[hyperp]
        #net = DeepSurv(shape_data, shape_label, param[0], param[1])
        net = pheNN(shape_data, shape_label, param[0], param[1])
    elif model == 2:
        net = LogisticRegression(shape_data, shape_label)
    elif model == 3:
        label_emb = np.load(f"../../data/Embedding/conv.npy", allow_pickle=True)
        parampair=[25,50,100,200,400,500,1000]
        hidden_size=parampair[hyperp]
        #net = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)
        label_emb=torch.tensor(label_emb,device=device).float()
        net = POPDxModelC(shape_data, shape_label, hidden_size, label_emb)
    else:
        raise
    nnnet = train(net, 1000, 10)
    whole_loader = DataLoader(testset, batch_size=len(testset))
    inputs, labels = next(iter(whole_loader))
    out = nnnet(inputs.to(device)).cpu().detach().numpy()
    out = torch.sigmoid(torch.from_numpy(out)).numpy()
    aucresult = []
    for i in range(labels.shape[1]):
        auc = renderresult(labels.cpu().detach().numpy()[:, i], out[:, i])
        aucresult.append(auc)
    loc='../../results/Disease_diagnosis/'
    try:
        os.mkdir(f"./{loc}pred")
    except:
        pass
    print(np.nanmean(aucresult))
    np.save(f"./{loc}pred/{category}{modelchar(model)}{image_X}_{hyperp}", aucresult)
    #np.save(f"./{loc}/{category}{modelchar(model)}{image_X}_{hyperp}out", out)
    #np.save(f"./{loc}/{category}{modelchar(model)}{image_X}_{hyperp}lab", labels)
    np.save(f"./{loc}pred/{image_X}lab", labels)
    torch.save(nnnet, f"./{loc}pred/{category}{modelchar(model)}{image_X}_{hyperp}model")
    print("complete")
