import sys
sys.path.append("../../disease_diagnosis")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from matplotlib import pyplot
from torch.autograd import Variable
from fileloader import load,loadindex
import torch.nn.functional as F
from survutil import (
    Survivaldata,
    ModelSaving,
    Cox,
    DeepSurv,
    EmbeddingModel,
    NegativeLogLikelihood,
)
trained_model = ''
survmodel = ''
d_dict_a = {
    'auc1':'^(I050|I051|I052|I060|I061|I062|I21(\d{0,3})?|I22(\d{0,3})?|I23(\d{0,3})?|I24[1-9]|I25(\d{0,3})?|I34[02]|I35[012]|I50(\d{0,3})?|I63|I64|G45)',
    'auc2':'^(I2[1-3]|^I241|^I252)',
    'auc3':'^N80',
    'auc4':'^I48',
    'auc5':'^I48|^I6[3-4]'
}
d_dict = {
    'infection': '^A|^B[0-8]',
    'Binfection': '^A[0-7]',
    'Vinfection': '^A[8-9]|^B[0-2]|^B3[0-4]',
    'cancer': '^C[0-8]|C9[0-7]',
    'lung_cancer': '^C34',
    'breast_cancer': '^C50',
    'prostate_cancer': '^C61',
    'leukaemia': '^C8[1-9]|^C9[0-6]',
    'BI_disorders': '^D[5-8]',
    'Anemia': '^D5|^D6[0-4]',
    'endocrine_disorders': '^E[0-8]|^E90',
    'diabetes': '^E1[0-4]',
    'obesity': '^E66',
    'MB_disorders': '^F',
    'dementia': '^F0[0-3]|^G30|^G31',
    'mood_disorders': '^F3[0-9]',
    'neurotic_disorders': '^F4[0-8]',
    'nervous_system_disorders': '^G',
    'parkinson_disease': '^G2[0-2]',
    'sleep_disorders': '^G47',
    'eye_disorders': '^H[0-5]',
    'ear_disorders': '^H[6-9]',
    'circulatory_system_disorders': '^I',
    'hypertension': '^I1[0-5]',
    'ischemic_heart_diseases': '^I2[0-5]',
    'arrhythmias': '^I4[6-9]',
    'heart_failure': '^I50',
    'stroke': '^I6[0-1]|^I6[3-4]',
    'peripheral_artery_diseases': '^I7[0-9]',
    'Respiratory_system_disorders': '^J',
    'COPD': '^J4[0-4]|^J47',
    'Astma': '^J4[5-6]',
    'Digestive_system_disorders': '^K',
    'inflammatory_bowel_diseases': '^K5[2-5]',
    'Liver_diseases': '^K7[0-7]',
    'skin_disorders': '^L',
    'musculoskeletal_system_disorders': '^M',
    'osteoarthritis': '^M1[5-9]',
    'genitourinary_system_disorders': '^N',
    'renal_failure': '^N1[7-9]'
}

df = pd.read_csv('../../data/phecode_icd10.csv')
input_file = pd.read_csv('../../data/phecode.csv')

def icd_to_phelogit(dname,type='pattern'):
    if type == 'pattern':
        PHECODE = df[df['ICD10'].str.contains(dname)]['PheCode'].unique()
    else:
        PHECODE = df[df['ICD10'].str.contains('|'.join(dname))]['PheCode'].unique()
    PHECODE = PHECODE[~np.isnan(PHECODE)]
    index = []
    for i in PHECODE:
        try:
            index.append(input_file[input_file['0'] == i].index.values[0])
        except:
            pass
    logits = np.zeros(1560)
    logits[index] = 1
    logits = logits.astype(bool)
    return logits

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
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
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


np.random.seed(0)
torch.manual_seed(0)

# auc
for key in d_dict_a.keys():
    raw={}
    print(key)
    device = torch.device("cuda")
    category = 6
    model = 1
    hyperp = 0
    image_X = 0
    input_logit = icd_to_phelogit(d_dict_a[key])
    Xdata, _,lab = load(image_X, category,inputcolumn=input_logit)
    # lab turn to (1,), when there's no 1 in the row, it will turn to 0, else 1
    net=torch.load(trained_model,map_location=device)
    net.outlayer=nn.Linear(in_features=net.outlayer.in_features, out_features=1, bias=True)
    numbers = list(range(lab.shape[0]))
    *_, trainindex, valindex, testindex = loadindex(image_X)
    learning_rate = 0.0001
    weight_decay = 0
    trainset = ukbdata(Xdata[trainindex], lab[trainindex])
    valset = ukbdata(Xdata[valindex], lab[valindex])
    testset = ukbdata(Xdata[testindex], lab[testindex])
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])

    nnnet = train(net, 100, 15)
    whole_loader = DataLoader(testset, batch_size=len(testset))
    inputs, labels = next(iter(whole_loader))
    out = nnnet(inputs.to(device)).cpu().detach().numpy()
    out = torch.sigmoid(torch.from_numpy(out)).numpy()
    aucresult = []
    
    auc = renderresult(labels.cpu().detach().numpy()[:], out[:])
    aucresult.append(auc)
    loc = key
    try:
        os.mkdir(f"./pred/{loc}")
    except:
        pass
    print(np.nanmean(aucresult))
    np.save(f"./pred/{loc}/{category}{modelchar(model)}{image_X}_{hyperp}", aucresult)
    #np.save(f"./pred/{loc}/{category}{modelchar(model)}{image_X}_{hyperp}out", out)
    #np.save(f"./pred/{loc}/{category}{modelchar(model)}{image_X}_{hyperp}lab", labels)
    np.save(f"./pred/{loc}/{image_X}lab", labels)
    torch.save(nnnet, f"./pred/{loc}/{category}{modelchar(model)}{image_X}_{hyperp}model")
    print("complete")

