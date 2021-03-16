import pandas as pd
import numpy as np
import os
import random
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from .utils import MyBar, colorizar, TorchBoard, headerizar

#=================== MODELS =============================================#

TRANS_NAME     = ""
FIXED_REP_SIZE = 90
# TOKENIZER_PRE = None
# TRANS_MODEL   = None

def setTransName(name:str):
    global TRANS_NAME
    TRANS_NAME = name

def setSeed(my_seed:int):
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)

# function that creates the transformer and tokenizer for later uses
def make_trans_pretrained_model(mod_only=False):
    '''
        This function return (tokenizer, model)
    '''
    tokenizer, model = None, None
    
    tokenizer = AutoTokenizer.from_pretrained(TRANS_NAME)
    model = AutoModel.from_pretrained(TRANS_NAME)

    if mod_only:
        return model 
    else:
        return tokenizer, model

# masqued mse loss used in multitask learning
class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, label):
        y_loss = self.mse(y_hat, y).reshape(-1)
        y_mask = label.reshape(-1)
        return (y_loss*y_mask).mean()

# Select a specific vector from a sequence
class POS(torch.nn.Module):
    def __init__(self, _p = 0):
        super(POS, self).__init__()
        self._p = _p
    def forward(self, X):
        return X[:,self._p]

# Be S = (v_0, ..., v_n ) a vector sequence, this return V = \sum^n_{i=0}v_i ~~ \frac{V}{|V|}
class ADDN(torch.nn.Module):
    def __init__(self):
        super(ADDN, self).__init__()
    def forward(self, X):
        return F.normalize(X.sum(dim=1), dim=-1)

# The encoder last layers
class Encod_Last_Layers(nn.Module):
    def __init__(self, hidden_size, vec_size):
        super(Encod_Last_Layers, self).__init__()
        self.mid_size = FIXED_REP_SIZE
        self.Dense1   = nn.Sequential(nn.Linear(vec_size, hidden_size), nn.LeakyReLU(),
                                      nn.Linear(hidden_size, self.mid_size), nn.LeakyReLU())
        # Classification
        self.Task1    = nn.Linear(self.mid_size, 2)
        # Regretion
        # self.Task2   = nn.Linear(self.mid_size, 1)

    def forward(self, X, ret_vec=False):
        y_hat = self.Dense1(X)
        if ret_vec:
            return y_hat
        y1 = self.Task1(y_hat).squeeze()
        # y2 = self.Task2(y_hat).squeeze()
        return y1 #, y2

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

# The encoder used in this work
class Encoder_Model(nn.Module):
    def __init__(self, hidden_size, vec_size=768, dropout=0.1, max_length=120, selection='first'):
        super(Encoder_Model, self).__init__()
        self.criterion1 = nn.CrossEntropyLoss()#weight=torch.Tensor([0.7,0.3]))
        # self.criterion2 = MaskedMSELoss()

        self.max_length = max_length
        self.tok, self.bert = make_trans_pretrained_model()

        self.encoder_last_layer = Encod_Last_Layers(hidden_size, vec_size)
        self.selection = None

        if selection   == 'addn':
            self.selection = ADDN()
        elif selection == 'first':
            self.selection = POS(0)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)
        
    def forward(self, X, ret_vec=False):
        ids   = self.tok(X, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)
        out   = self.bert(**ids)
        vects = self.selection(out[0])
        return self.encoder_last_layer(vects, ret_vec=ret_vec)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path) 
    
    def makeOptimizer(self, lr=5e-5, lr_factor=9/10, decay=2e-5, algorithm='adam'):
        pars = [{'params':self.encoder_last_layer.parameters()}]

        for l in self.bert.encoder.layer:
            lr *= lr_factor
            D = {'params':l.parameters(), 'lr':lr}
            pars.append(D)
        try:
            lr *= lr_factor
            D = {'params':self.bert.pooler.parameters(), 'lr':lr}
            pars.append(D)
        except:
            print('#Warning: Pooler layer not found')

        if algorithm == 'adam':
            return torch.optim.Adam(pars, lr=lr, weight_decay=decay)
        elif algorithm == 'rms':
            return torch.optim.RMSprop(pars, lr=lr, weight_decay=decay)

# The model's maker
def makeModels(name:str, hidden_size, _tr_vec_size=768, dropout=0.0, max_length=120, selection='first'):
    if name == 'encoder':
        return Encoder_Model(hidden_size, _tr_vec_size, dropout=dropout, max_length=max_length, selection=selection)
    else:
        print('# The models name', headerizar(name), 'is invalid, instead use one of this: [encoder]')
        exit(-1)
        return None

def trainModels(model, Data_loader, epochs:int, evalData_loader=None, lr=0.1, etha=1., nameu='encoder', optim=None, b_fun=None, smood=False, mtl=False, use_acc=True):
    if epochs <= 0:
        return
    if optim is None:
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    board = TorchBoard()
    if b_fun is not None:
        board.setFunct(b_fun)
    
    for e in range(epochs):
        bar = MyBar('Epoch '+str(e+1)+' '*(int(math.log10(epochs)+1) - int(math.log10(e+1)+1)) , 
                    max=len(Data_loader)+(len(evalData_loader if evalData_loader is not None else 0)))
        total_loss, total_acc, dl = 0., 0., 0
        for data in Data_loader:
            optim.zero_grad()
            
            # Multi-Task learning for now not
            # y_hat, y_val = model(data['x'])
            y_hat = model(data['x'])
            y1    = data['y'].to(device=model.device)
            # y2 = data['v'].float().to(device=model.device)
            
            if mtl:
                l1 = model.criterion1(y_hat, y1)
                l2 = model.criterion2(y_val, y2, y1)
                loss = etha*l1 + (1. - etha)*l2
            else:
                loss = model.criterion1(y_hat, y1)
            
            loss.backward()
            optim.step()

            with torch.no_grad():
                total_loss += loss.item() * y1.shape[0]
                total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
                dl += y1.shape[0]
            bar.next(total_loss/dl)
        if use_acc:
            res = board.update('train', total_acc/dl, getBest=True)
        else:
            res = board.update('train', total_loss/dl, getBest=True)
        
        # Evaluate the model
        if evalData_loader is not None:
            total_loss, total_acc, dl= 0,0,0
            with torch.no_grad():
                for data in evalData_loader:
                    # y_hat, y_val = model(data['x'])
                    y_hat = model(data['x'])
                    y1 = data['y'].to(device=model.device)
                    # y2 = data['v'].to(device=model.device)

                    if mtl:
                        l1 = model.criterion1(y_hat, y1)
                        l2 = model.criterion2(y_val, y2, y1)
                        loss = etha*l1 + (1. - etha)*l2
                    else:
                        loss = model.criterion1(y_hat, y1)
                    
                    total_loss += loss.item() * y1.shape[0]
                    total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
                    dl += y1.shape[0]
                    bar.next()
            if use_acc:
                res = board.update('test', total_acc/dl, getBest=True)
            else:
                res = board.update('test', total_loss/dl, getBest=True)
        bar.finish()
        del bar
        
        if res:
            model.save(os.path.join('pts', nameu+'.pt'))
    board.show(os.path.join('pts', nameu+'.png'), plot_smood=smood)

def evaluateModels(model, testData_loader, header=('id', 'is_humor', 'humor_rating'), cleaner=[], name='pred'):
    model.eval()
    
    pred_path = os.path.join('out', name+'.csv')

    bar = MyBar('test', max=len(testData_loader))
    Ids, lab, val = [], [], []
    
    cpu0 = torch.device("cpu")
    # with torch.no_grad():
    #     for data in testData_loader:
    #         y_hat, y_val = model(data['x'])
    #         y_hat, y_val = y_hat.to(device=cpu0), y_val.to(device=cpu0)
            
    #         y_hat = y_hat.argmax(dim=-1).squeeze()
    #         y_val = y_val.squeeze() * y_hat
    #         ids = data['id'].squeeze()
            
    #         for i in range(ids.shape[0]):
    #             Ids.append(ids[i].item())
    #             lab.append(y_hat[i].item())
    #             val.append(y_val[i].item())
    #         bar.next()
    # bar.finish()

    # Ids, lab, val = pd.Series(Ids), pd.Series(lab), pd.Series(val)
    # data = pd.concat([Ids, lab, val], axis=1)
    # del Ids
    # del lab
    # del val
    # data.to_csv(pred_path, index=None, header=header)
    # del data
    # print ('# Predictions saved in', colorizar(pred_path))
    
    # if len(cleaner) > 0:
    #     data = pd.read_csv(pred_path)
    #     data.drop(cleaner, axis=1, inplace=True)
    #     data.to_csv(pred_path, index=None)
    #     print ('# Cleaned from', ', '.join(cleaner) + '.')

    with torch.no_grad():
        for data in testData_loader:
            y_hat = model(data['x'])
            y_hat = y_hat.to(device=cpu0)
            
            y_hat = y_hat.argmax(dim=-1).squeeze()
            ids = data['id'].squeeze()
            
            for i in range(ids.shape[0]):
                Ids.append(ids[i].item())
                lab.append(y_hat[i].item())
            bar.next()
    bar.finish()
    
    Ids, lab = pd.Series(Ids), pd.Series(lab)
    data = pd.concat([Ids, lab], axis=1)
    del Ids
    del lab
    data.to_csv(pred_path, index=None, header=header)
    del data
    print ('# Predictions saved in', colorizar(pred_path))
    
    if len(cleaner) > 0:
        data = pd.read_csv(pred_path)
        data.drop(cleaner, axis=1, inplace=True)
        data.to_csv(pred_path, index=None)
        print ('# Cleaned from', ', '.join(cleaner) + '.')

#=================== DATASETS =============================================#

class RawDataset(Dataset):
    def __init__(self, csv_file, id_h='id', text_h='text', class_h='is_humor'):
        self.data_frame = pd.read_csv(csv_file)
        self.x_name  = text_h
        self.id_name = id_h
        self.y1_name = class_h

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ids  = self.data_frame.loc[idx, self.id_name]
        sent  = self.data_frame.loc[idx, self.x_name]
        
        try:
            y1 = self.data_frame.loc[idx, self.y1_name]
            # regv  = self.data_frame.loc[idx, 'humor_rating'] if int(value) != 0 else 0.
        except:
            y1 = 0
            # value, regv = 0, 0.

        # sample = {'x': sent, 'y': value, 'v':regv, 'id':ids}
        sample = {'x': sent, 'y': y1, 'id':ids}
        return sample

def makeDataSet(csv_path:str, batch, shuffle=True, id_h='id', text_h='text', class_h='is_humor'):
    data   =  RawDataset(csv_path, id_h=id_h, text_h=text_h, class_h=class_h)
    loader =  DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=4, drop_last=False)
    return data, loader

#================= TEMPORAL FUNCTIONS ===============================

def makeTrain_and_ValData(data_path:str, percent=10, class_label=None, df='data'):
	'''
		class_lable: str The label to split, the humor column with values ['0', '1']
	'''
	train_path = os.path.join(df, 'train_data.csv')
	eval_path  = os.path.join(df, 'eval_data.csv')

	if os.path.isfile(train_path) and os.path.isfile(eval_path):
		return train_path, eval_path	

	data = pd.read_csv(data_path)	
	mean = [len(data.loc[i, 'text'].split()) for i in range(len(data))]
	var  = [i*i for i in mean]
	mean, var = sum(mean)/len(mean), sum(var)/len(mean)
	var = (var - mean) ** 0.5
	print ('# Mean:', mean, 'std:', var)

	
	train_data, eval_data = None, None
	if class_label is None:
		percent = (len(data) * percent) // 100
		ides = [i for i in range(len(data))]
		random.shuffle(ides)

		train_data = data.drop(ides[:percent])
		eval_data  = data.drop(ides[percent:])
	else:
		pos  = list(data.query(class_label+' == 1').index)
		neg  = list(data.query(class_label+' == 0').index)
		random.shuffle(pos)
		random.shuffle(neg)

		p1,p2 = (len(pos) * percent) // 100, (len(neg) * percent) // 100
		indes_t, indes_e = pos[:p1] + neg[:p2], pos[p1:] + neg[p2:]

		train_data = data.drop(indes_t)
		eval_data  = data.drop(indes_e)

	train_data.to_csv(train_path, index=None)
	eval_data.to_csv(eval_path, index=None)

	return train_path, eval_path

def convert2EncoderVec(data_name:str, model, loader, save_pickle=False, save_as_numpy=False, df='data'):
    model.eval()
    IDs, YC, YV, X = [], [], [], []

    new_name = os.path.join('data', data_name+'.csv' if not save_pickle else data_name+'.pkl')

    print ('# Creating', colorizar(os.path.basename(new_name)))
    bar = MyBar('change', max=len(loader))

    cpu0 = torch.device("cpu")
    with torch.no_grad():
        for data in loader:
            x = model(data['x'], ret_vec=True).to(device=cpu0).numpy()
            try:
                y_c = data['y']
            except:
                y_c = None
            
            try:
                y_v = data['v']
            except:
                y_v = None
            
            try:
                ids = data['id']
            except:
                ids = None

            for i in range(x.shape[0]):
                l = x[i,:].tolist()
                X.append(' '.join([str(v) for v in l]))
                if y_c is not None:
                    YC.append(int(y_c[i].item()))
                if y_v is not None:
                    YV.append(y_v[i].item())
                if ids is not None:
                    IDs.append(ids[i].item())
            bar.next()
    bar.finish()

    if save_as_numpy:
        X_t = [v for v in map(lambda x: [float(s) for s in x.split()], X)]
        X_t = np.array(X_t, dtype=np.float32)
        np.save(os.path.join('data', data_name+'.npy'), X_t)
        del X_t
        
        if len(IDs) > 0:
            ids_t = np.array([int(i) for i in IDs], dtype=np.int64)
            np.save(os.path.join('data', data_name+'_id.npy'), ids_t)
            del ids_t

    conca, n_head = [], []
    if len(IDs) > 0:
        conca.append(pd.Series(IDs))
        n_head.append('id')
        del IDs
    if len(YC) > 0:
        conca.append(pd.Series(YC))
        n_head.append('is_humor')
        del YC
    if len(YV) > 0:
        conca.append(pd.Series(YV))
        n_head.append('humor_rating')
        del YV
    conca.append(pd.Series(X))
    n_head.append('vecs')
    del X

    data = pd.concat(conca, axis=1)
    if save_pickle:
        data.to_pickle(new_name)
    else:
        data.to_csv(new_name, index=None, header=n_head)
    return new_name
