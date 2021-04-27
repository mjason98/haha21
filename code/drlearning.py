import torch 
import torch.nn.functional as F
import math
import numpy as np
from random import shuffle

class ExperienceReplay:
    def __init__(self, N=250, batch_size=64):
        self.N = N 
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0

        self.device_cpu = torch.device("cpu")
    def add_memory(self, state1, action, reward, state2):
        self.counter += 1
        if self.counter % self.N == 0:
            self.shuffle_memory()
        
        if len(self.memory) < self.N:
            self.memory.append((state1.detach().to(device=self.device_cpu), action, reward, state2.detach().to(device=self.device_cpu)))
        else:
            rand_index = np.random.randint(0,self.N-1)
            self.memory[rand_index] = (state1.detach().to(device=self.device_cpu), action, reward, state2.detach().to(device=self.device_cpu))
    
    def shuffle_memory(self):
        shuffle(self.memory)
    def get_batch(self):
        with torch.no_grad():
            if len(self.memory)  < self.batch_size:
                batch_size = len(self.memory)
            else:
                batch_size = self.batch_size
            
            if len(self.memory) < 1:
                raise IndexError("ERROR: No data in memory")
            
            ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
            batch = [self.memory[i] for i in ind]
            state1_batch = torch.stack( [x[0].squeeze(dim=0) for x in batch], dim=0)
            action_batch = torch.Tensor([x[1] for x in batch]).long()
            reward_batch = torch.Tensor([x[2] for x in batch])
            state2_batch = torch.stack( [x[3].squeeze(dim=0) for x in batch], dim=0)
        return state1_batch, action_batch, reward_batch, state2_batch

class PositionalEncoding(torch.nn.Module):
    ''' based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html'''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(1000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0,1)
        # self.register_buffer('pe',pe)
        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.pe.to(device=self.device)
        # self.to(device=self.device)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def policy(qvalues, eps=None):
    with torch.no_grad():
        if eps is not None:
            if torch.rand(1) < eps:
                return torch.randint(low=0,high=12, size=(1,))
            else:
                return torch.argmax(qvalues)
        else:
            return torch.multinomial(F.softmax(F.normalize(qvalues), dim=0), num_samples=1)

class Agent_DQL(torch.nn.Module):
    def __init__(self, naction, nhead=3, nhid=128, d_model=90, nlayers=3, max_len=5000, dropout=0.1):
        super(Agent_DQL, self).__init__()
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, naction)
        self.init_weights()

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)
        self.pos_encoder.pe.to(device=self.device)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self, x_):
        x = x_.to(device=self.device)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        y = x[:,-1,:] # last one
        y = self.decoder(y)
        return y
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path) 

class Phi(torch.nn.Module): # encoder net
    def __init__(self, d_model, max_len, nhead=1, nlayers=1, hiden_size=100, dropout=0.1):
        super(Phi, self).__init__()
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.model_type = 'Transformer_Phi'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, hiden_size, dropout) # 1 nhead
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # 1 nlayers

        self.d_model = d_model

        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.to(device=self.device)

    def forward(self, x_):
        x = x_ * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        y = x.flatten(start_dim=1)
        return y

class Gnet(torch.nn.Module): # inverse model
    def __init__(self, actions, size):
        super(Gnet, self).__init__()
        self.linear1 = torch.nn.Linear(size*2,(size*2+actions)//2)
        self.linear2 = torch.nn.Linear((size*2+actions)//2, actions)

        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.to(device=self.device)

    def forward(self, state1, state2):
        x = torch.cat((state1, state2), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y,dim=1)
        return y

class Fnet(torch.nn.Module): # forward model
    def __init__(self, actions, size):
        super(Fnet, self).__init__()
        v1 = (size + actions + size) //2
        v2 =  (v1 + size) // 2
        self.linear1 = torch.nn.Linear(size + actions, v1)
        self.linear2 = torch.nn.Linear(v1,v2)
        self.linear3 = torch.nn.Linear(v2,size)
        self.actions = actions
        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.to(device=self.device)

    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0],self.actions) # converta actions to one hot
        indices = torch.stack((torch.arange(action.shape[0]), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1. 

        # x = torch.cat((state.to(device=self.device),action_.to(device=self.device)), dim=1)
        x = torch.cat((state,action_), dim=1)
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        y = self.linear3(y)
        return y

class ICM(torch.nn.Module):
    def __init__(self, actions, size, d_model, max_len=5000, forward_scale=1., inverse_scale=1e4,  nhead=1, nlayers=1, hiden_size=128, dropout=0.1):
        super(ICM, self).__init__()

        self.encoder = Phi(d_model, max_len, nhead=nhead, nlayers=nlayers, hiden_size=hiden_size, dropout=dropout)
        self.forward_model = Fnet(actions, size)
        self.inverse_model = Gnet(actions, size)
        self.forward_scale = forward_scale
        self.inverse_scale = inverse_scale

        self.forward_loss = torch.nn.MSELoss(reduction='none')
        self.inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)
        self.encoder.pos_encoder.pe.to(device=self.device)
    
    def forward(self, state1, action, state2):
        state1_hat = self.encoder(state1).to(device=self.device)
        state2_hat = self.encoder(state2).to(device=self.device)

        state2_hat_pred = self.forward_model(state1_hat.detach(), action.detach())
        forward_pred_err = self.forward_scale * self.forward_loss(state2_hat_pred, state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        
        pred_action = self.inverse_model(state1_hat, state2_hat)
        inverse_pred_err = self.inverse_scale * self.inverse_loss(pred_action, action.detach().flatten()).unsqueeze(dim=1)

        return forward_pred_err, inverse_pred_err
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path) 