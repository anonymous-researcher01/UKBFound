from config import *
import pandas as pd
from survutil import (
    Survivaldata,
    ModelSaving,
    NegativeLogLikelihood,
)
from torch.utils.data import DataLoader
from survutil import  c_index
from fileloader import load, loadindex
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
gpu = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
save_path = '../../results/ComparativeAnalysis/'
net_loc = '../../results/Disease_diagnosis/surv/610_0model'
df = pd.read_csv('../../data/phecode_icd10.csv')
input_file = pd.read_csv('../../data/phecode.csv')
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
d_dict_a = {
    'auc1':'^(I050|I051|I052|I060|I061|I062|I21(\d{0,3})?|I22(\d{0,3})?|I23(\d{0,3})?|I24[1-9]|I25(\d{0,3})?|I34[02]|I35[012]|I50(\d{0,3})?|I63|I64|G45)',
    'auc2':'^(I2[1-3]|^I241|^I252)',
    'auc3':'^N80',
    'auc4':'^I48',
    'auc5':'^I48|^I6[3-4]'
}
d_dict_c = {
    'cindex2chd':'^(I20|I21|I22|I23|I24|I252|I25|I46|R96|R98|Z951|T822)',
    'cindex2af':'^I48',
    'cindex2t2d':'E1[0-1]',
    'cindex2bc':'C50',
    'cindex2pc':'C61',
    'cindex3':'^(G45|I11|I13|I2[0-5]|I42|I46|I50|I6[0-7]|I71|I72|I74)',
    'cindex4CVD':'^(I2[1-5]|I6)',
    'cindex4CHD':'^I2[1-5]',
    'cindex4stroke':'I6[0-9]'
}

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
        early_break(np.mean(val_losses))

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

def modelchar(x):
    if x >= 0 and x <= 9:
        return str(x)
    elif x >= 10:
        return chr(65 + x - 10)


def form(dir):
    auc={}
    for i in os.listdir(dir):
        if '5w' not in i:
            cat=i.split('_')[1]
            try:
                d=np.load(dir+i,allow_pickle=True)
            except:
                d=[0,0,np.nan]
            if cat not in auc.keys():
                auc[cat]=[d[2]]
            else:
                auc[cat].append(d[2])
    return auc


def icd_to_phelogit(dname,type='pattern'):
    if type == 'pattern':
        PHECODE = df[df['ICD10'].str.contains(dname)]['PheCode'].unique()
    else:
        PHECODE = df[df['ICD10'].str.contains('|'.join(dname))]['PheCode'].unique()
    PHECODE = PHECODE[~np.isnan(PHECODE)]
    # find the index of each PHECODE in the input_file 1st column
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

np.random.seed(0)
torch.manual_seed(0)

for key in d_dict.keys():
    raw={}
    print(key)
    device = torch.device("cuda")
    net=torch.load(net_loc,map_location=device)
    category = 6
    model = 1
    hyperp = 0
    image_X = 0
    input_logit = icd_to_phelogit(d_dict[key])
    net.outlayer=nn.Linear(in_features=net.outlayer.in_features, out_features=1, bias=True)
    learning_rate = 0.005
    X, Y, E = load(image_X=image_X, category=category, only=False,inputcolumn=input_logit)
    index = np.isnan(E)
    E = np.where(index, torch.tensor(0), E)
    numbers = list(range(X.shape[0]))
    *_, trainindex, valindex, testindex = loadindex(image_X)
    trainset = Survivaldata(X, Y, E, trainindex)
    valset = Survivaldata(X, Y, E, valindex)
    testset = Survivaldata(X, Y, E, valindex)
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])
   
    nnet = train(net, 100,15, train_loader, val_loader, device, learning_rate)
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
    print(np.nanmean(cindex))
    loc=key
    try:
        os.mkdir(f"./surv/{loc}")
    except:
        pass
    np.save(f"./surv/{loc}/{category}{modelchar(model)}{image_X}_{hyperp}", cindex)
    np.save(f"./surv/{loc}/{category}{modelchar(model)}{image_X}out", out)
    #np.save(f"./surv/{loc}/{image_X}lab", [ytest, etest])
    torch.save(nnet, f"./surv/{loc}/{category}{modelchar(model)}{image_X}_{hyperp}model")
    print("complete")