import time
import random
import pandas as pd
import multiprocessing as mp 
import numpy as np
import os 
import torch
import torch.nn.functional as F
import copy

from .utils import strToListF
from .models import makeDataSet_Vec
from .utils import strToListF, colorizar

# models
from .drlearning import Agent_DQL, ExperienceReplay, ICM as ICM_DQL

class VecDataEnvironment:
    ''' If this environment return done=True, reset it or some errors may apears'''
    VERY_BAD_REWARD = -1.
    def __init__(self, data_path, eval_path=None, max_backpack_size=200, vname='vecs', lname='is_humor', frmethod='acc', eval_data_path=None):
        self.data = pd.read_csv(data_path)
        self.data_eval = None
        if eval_path is not None:
            self.data_eval  = pd.read_csv(eval_path)
            # self.min_dist   = np.zeros(len(self.data_eval), dtype=np.float32)
            # self.assg_label = [None for _ in range(len(self.data_eval))]

        self.max_backpack_size = max_backpack_size
        self.vec_size = len(self.data.loc[0,vname].split())
        self.vname = vname 
        self.lname = lname

        self.done = False 
        self.backpack = []
        self.backpack_l = []

        self.pos_gone = None
        self.iterator = None
        self.iter_pos = None
        self.current_vector = None
        self.final_reward = None
        self.frmethod = frmethod
    
    def __next(self):
        if self.iterator is None:
            self.iterator = [i for i in range(len(self.data))]
            self.iter_pos = 0
        
        if self.iter_pos >= len(self.iterator):
            self.done = True
            self.__calculate_final_R()
            return None, None

        i = self.iterator[self.iter_pos]
        self.iter_pos += 1

        cad = strToListF(self.data.loc[i, self.vname])
        lab = int(self.data.loc[i, self.lname]) 
        return cad, lab    
    
    def export_prototypes(self, file_list, label_list):
        ''' export to a txt the vectors in the backpak\n
            filelist: [f1:Str, f2:str, ... , fn:str] \n 
            label_list: [l1:int, l2:int, ..., ln:int] \n 
            the vectors with li label will be placed in fi file '''
        for file_, label_ in zip(file_list, label_list):
            print ('# Exporting prototypes to', colorizar(os.path.basename(file_)))
            with open(file_, 'w') as file:
                for v,l in zip(self.backpack, self.backpack_l):
                    if l != label_: continue
                    chad = ' '.join([str(v) for v in v.tolist()]) + '\n'
                    file.write(chad)
    
    def proto_cmp_data_csv(self, ini_fin):
        ''' function used with paralellism to calculate the labels of the data with the prototypes.\n
            ini_fin:pair (ini:int, fin:int) the initial and final position of data, accesed by data.loc[i, vname] for i in [ini,fin) '''
        sol = []
        for i in range(ini_fin[0], ini_fin[1]):
            # lab   = int(data.loc[i, lname])
            vec = None 
            if self.data_eval is not None:
                vec = np.array(strToListF(self.data_eval.loc[i, self.vname]), dtype=np.float32)
            else: 
                vec = np.array(strToListF(self.data.loc[i, self.vname]), dtype=np.float32) 
            min_val, l_min = None, None
            for v, l in zip(self.backpack, self.backpack_l):
                if l is None : continue
                # euclidiean distance
                current_value = np.sqrt(((v - vec) ** 2).sum())
                if min_val is None or min_val > current_value:
                    min_val = current_value
                    l_min   = l 
            if l_min is None:
                break
            sol.append(l_min)
        return np.array(sol, np.int32) # check this later, the int32 ------------------------------------------ OJO -----------------
    
    # def proto_eval_minD(self, ini_fin):
    #     ''' This function uses paralelism to calculate the min distances '''
    #     sol = 0.
    #     current_v = np.array(self.current_vector[0], dtype=np.float32)
    #     for i in range(ini_fin[0], ini_fin[1]):
    #         vec = np.array(strToListF(self.data_eval.loc[i, self.vname]), dtype=np.float32)        
    #         dist_in = np.sqrt(((vec - current_v) ** 2).sum())

    #         if (self.min_dist[i] > dist_in) or (self.assg_label[i] is None):
    #             self.min_dist[i] = dist_in
    #             if (self.assg_label[i] is None) or (self.assg_label[i] != self.current_vector[1]):
    #                 labu = int(self.data_eval.loc[i, self.lname])
    #                 if    labu == self.current_vector[1] : sol += 1.
    #                 else: sol += -1.
    #             self.assg_label[i] = self.current_vector[1]
    #     # es mejor poner estas variables like globales y no asi, compartidas
    #     return sol, self.min_dist[ini_fin[0]:ini_fin[1]], self.assg_label[ini_fin[0]:ini_fin[1]]
    
    # def proto_gone_minD(self, ini_fin):
    #     ''' This function uses paralelism to calculate the min distances '''
    #     sol, eps = 0., 1e-9
    #     if self.pos_gone is not None:
    #         current_v = self.backpack[ self.pos_gone ]
    #         for i in range(ini_fin[0], ini_fin[1]):
    #             vec = np.array(strToListF(self.data_eval.loc[i, self.vname]), dtype=np.float32)        
    #             dist_out = np.sqrt(((vec - current_v) ** 2).sum()).item()

    #             if (self.min_dist[i] - dist_out) < eps and (self.assg_label[i] is not None):
    #                 self.min_dist[i] = None
    #                 old_asl =  self.assg_label[i]
    #                 labu = int(self.data_eval.loc[i, self.lname])
    #                 for ide, (bv, bl) in enumerate(zip(self.backpack, self.backpack_l)):
    #                     if ide == i: continue
    #                     if bl is None: continue
                        
    #                     d2 = np.sqrt(((vec - bv) ** 2).sum()).item()
    #                     if (self.min_dist[i] is None) or (d2 < self.min_dist[i]):
    #                         self.min_dist[i] = d2
    #                         self.assg_label[i] = bl
    #                 if old_asl != self.assg_label[i]:
    #                     if    labu == self.assg_label[i]: sol += 1.
    #                     else: sol -= 1.

    #     return sol, self.min_dist[ini_fin[0]:ini_fin[1]], self.assg_label[ini_fin[0]:ini_fin[1]]

    def __calculate_final_R(self):
        ''' Inside this, self.iterator is seted to None, be aware of future errors '''
        cnt = mp.cpu_count()
        pool = mp.Pool(cnt)

        if self.data_eval is not None:
            dx = int(len(self.data_eval) / cnt ) 
            dx = [(i*dx, i*dx + dx + (0 if i != cnt-1 else len(self.data_eval) % cnt)) for i in range(cnt)]
        else:
            dx = int(len(self.data) / cnt ) 
            dx = [(i*dx, i*dx + dx + (0 if i != cnt-1 else len(self.data) % cnt)) for i in range(cnt)]
        
        label_list = pool.map(self.proto_cmp_data_csv, dx)
        del pool 
        label_list = np.concatenate(label_list, axis=0)

        if label_list.shape[0] <= 0:
            # The backpack is empty !
            self.final_reward = self.VERY_BAD_REWARD
            return
        
        if self.data_eval is not None:
            original_label = np.array(self.data_eval[self.lname].tolist(), dtype=np.int32)
        else:
            original_label = np.array(self.data[self.lname].tolist(), dtype=np.int32)
        if self.frmethod == 'acc':
            self.final_reward = ((label_list == original_label).sum() / original_label.shape[0]).item()
        del label_list 
        del original_label

    def __reset_backpack(self):       
        if len(self.backpack) <= 0:
            for _ in range(self.max_backpack_size):
                self.backpack.append(np.array([0 for _ in range(self.vec_size)], dtype=np.float32))
                self.backpack_l.append(None)
        else:
            for k in range(self.max_backpack_size):
                self.backpack[k] = np.array([0 for _ in range(self.vec_size)], dtype=np.float32)
                self.backpack_l[k] = None
        # if self.data_eval is not None:
        #     self.min_dist -= self.min_dist
        #     for i in range(len(self.assg_label)): self.assg_label[i] = None

    def __makeState(self):
        self.current_vector = self.__next()
        backP = np.stack(self.backpack, axis=0)
        if self.done:
            vecI = np.zeros(self.vec_size, dtype=np.float32)
        else:
            vecI  = np.array(self.current_vector[0], dtype=np.float32)
        return (backP, vecI)
    
    def __calculate_in_reward(self):
        # por ahora 
        return 0.
        # if self.data_eval is None:
        #     return 0
        # cnt = mp.cpu_count()
        # pool = mp.Pool(cnt)        
        
        # dx = int(len(self.data_eval) / cnt ) 
        # dx = [(i*dx, i*dx + dx + (0 if i != cnt-1 else len(self.data_eval) % cnt)) for i in range(cnt)]
        # in_reward, out_reward = 0., 0.

        # mdist_list = pool.map(self.proto_eval_minD, dx)
        
        # # esto se puede mejorar con memoria compartida, pero ahora me pesa
        # new_di, new_ass = [], []
        # for v,d,l in mdist_list: 
        #     in_reward += v
        #     new_di.append(d)
        #     new_ass += l 
        
        # self.min_dist = np.concatenate(new_di)
        # self.assg_label[:] = new_ass
        
        # mdist_list = pool.map(self.proto_gone_minD, dx)
        # new_di, new_ass = [], []
        # for v,d,l in mdist_list: 
        #     out_reward += v
        #     new_di.append(d)
        #     new_ass += l 
        
        # self.min_dist = np.concatenate(new_di)
        # self.assg_label[:] = new_ass

        # del mdist_list
        # del pool         
        # return in_reward + out_reward

    def reset(self):
        ''' Return the pair: a np.array of shape (max_backpack_size, vec_size) and a np.array of shape (vec_size). 

        They are: (backpack state, incoming vector from data). '''
        self.done = False 
        self.final_reward = None
        self.iterator = None
        self.__reset_backpack()
        s,v = self.__makeState()
        return s,v
    
    def step(self, action:int):
        ''' Return four objects: \n
            \t BackPack State, Incoming Vector from data, reward, done \n
            \t types: np.array(max_backpack_size, vec_size) , np.array (vec_size), float, bool '''
        if action < 0 or action > self.max_backpack_size:
            raise ValueError('ERROR in action input variable, action: {} not in [0,{}]'.format(action,self.max_backpack_size))
        
        self.pos_gone = action if action < self.max_backpack_size else None 
        reward = self.__calculate_in_reward()

        if action < self.max_backpack_size:
            self.backpack[action] = np.array(self.current_vector[0], dtype=np.float32)
            self.backpack_l[action] = int(self.current_vector[1])

        s,v = self.__makeState()
        if self.final_reward is not None: reward += self.final_reward
        
        return s, v, reward, self.done

def prepareBackpackState(blist, vec):
    state = np.concatenate([blist,vec.reshape(1,-1)], axis=0)
    state = torch.from_numpy(state)
    return state


def __policy_dql(qvalues, nactions=12,eps=None):
    with torch.no_grad():
        if eps is not None:
            if torch.rand(1) < eps:
                return torch.randint(low=0,high=nactions, size=(1,))
            else:
                return torch.argmax(qvalues)
        else:
            return torch.multinomial(F.softmax(F.normalize(qvalues), dim=0), num_samples=1)

def __minibatch_train_dql(Qmodel, Qtarget, qloss, replay, params, DEVICE, icm=None):
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1).to(device=DEVICE)
    reward_batch = reward_batch.view(reward_batch.shape[0],1).to(device=DEVICE)

    state1_batch = state1_batch.to(device=DEVICE)
    state2_batch = state2_batch.to(device=DEVICE)

    forward_pred_err , inverse_pred_err = 0., 0.
    if icm is not None:
        forward_pred_err , inverse_pred_err = icm(state1_batch, action_batch, state2_batch)
    
    i_reward = (1. / float(params['eta'])) * forward_pred_err
    reward = i_reward.detach()
    if bool(params['intrinsic']):
        reward += reward_batch
    
    # qvals = Qmodel(state2_batch) # recordar usar target net later
    qvals = Qtarget(state2_batch)
    reward += float(params['gamma']) * torch.max(qvals)
    
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack((torch.arange(action_batch.shape[0]),action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))

    return forward_pred_err, inverse_pred_err, q_loss

def __loss_fn(q_loss, inverse_loss, forward_loss, params):
    loss_  = (1 - float(params['beta'])) * inverse_loss 
    loss_ += float(params['beta']) * forward_loss
    loss_ = loss_.mean() # loss_.sum() / loss.flatten().shape[0]
    loss = loss_ + float(params['lambda']) * q_loss
    return loss 

# params (data_path, lcolumn, vcolumn, param)
def __prototypes_with_dql(params):
    print ('# Start:','Deep Q Learning algorithm, relax, this will take a wille.')
    BACKPACK_SIZE, EPS  = int(params['max_prototypes']),  float(params['eps'])
    EPOCHS, LR, BSIZE = int(params['epochs']), float(params['lr']), int(params['batch_size'])
    DMODEL = params['d_model']
    target_refill, i_targetFill = 100, 0

    losses = []
    switch_to_eps_greedy = int(EPOCHS * (2/5))
    
    env = VecDataEnvironment(params['data_path'], eval_path=params['eval_data_path'], max_backpack_size=BACKPACK_SIZE, vname=params['vcolumn'], lname=params['lcolumn'])
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # max_len : 5000, antes BACKPACK_SIZE+11, de esta forma quisas se adapte a ir cresiendo poco a poco
    Qmodel = Agent_DQL(BACKPACK_SIZE+1, nhead=params['nhead'],nhid=params['nhid'],d_model=DMODEL,nlayers=params['n_layers'], max_len=5000,dropout=params['dropout'])
    qloss = torch.nn.MSELoss().to(device=DEVICE)
    
    # seting up the taget net, and memory replay stuff
    Qtarget = copy.deepcopy(Qmodel).to(device=DEVICE)
    Qtarget.load_state_dict(Qmodel.state_dict())
    replay = ExperienceReplay(N=int(params['memory_size']), batch_size=BSIZE)

    icm = ICM_DQL(BACKPACK_SIZE+1, DMODEL*(BACKPACK_SIZE+1), DMODEL, max_len=5000, forward_scale=1., inverse_scale=1e4, nhead=params['nhead'],hiden_size=params['nhid'],nlayers=params['n_layers'], dropout=params['dropout'])
    all_model_params = list(Qmodel.parameters()) + list(icm.parameters())
    opt = torch.optim.Adam(lr=LR, params=all_model_params)
    
    Qmodel.train()
    icm.train()

    greater_reward = -(2**30)
    for i in range(EPOCHS):
        print('# Epoch', i+1, 'with eps' if i >= switch_to_eps_greedy else 'with softmax policy')

        all_obj_seeit = False
        state1 = prepareBackpackState(*env.reset()).unsqueeze(0).to(device=DEVICE)
        acc_reward = 0.
        it_episode = 0
        init_time = time.time()
        while not all_obj_seeit:
            # parafernalia ----------------------------
            it_episode += 1
            print ('\r  It {} with reward {:.4f} dt:{:.1f}'.format(it_episode, acc_reward, time.time()-init_time), end=' ')
            # -----------------------------------------
            
            opt.zero_grad()
            q_val_pred = Qmodel(state1)
        
            # Use softmax policy only at the begining
            if i >= switch_to_eps_greedy:
                action = int(__policy_dql(q_val_pred, nactions=BACKPACK_SIZE+1,eps=EPS))
            else:
                action = int(__policy_dql(q_val_pred, nactions=BACKPACK_SIZE+1))

            back_state, vec_state , e_reward, done = env.step(action)
            state2 = prepareBackpackState(back_state, vec_state).unsqueeze(0).to(device=DEVICE)
            
            replay.add_memory(state1, action, e_reward, state2)
            acc_reward += e_reward
            
            all_obj_seeit = done
            if not done:
                state1 = state2
            
            if len(replay.memory) < BSIZE:
                continue
            forward_pred_err, inverse_pred_err, q_loss = __minibatch_train_dql(Qmodel, Qtarget, qloss, replay, params, DEVICE, icm=icm)
        
            loss = __loss_fn(q_loss, forward_pred_err, inverse_pred_err, params)
            loss_list = (q_loss.mean().item(), forward_pred_err.flatten().mean().item(), inverse_pred_err.flatten().mean().item())
            losses.append(loss_list)
            loss.backward()
            opt.step()

            i_targetFill += 1
            if i_targetFill % target_refill == 0:
                i_targetFill = 0
                Qtarget.load_state_dict(Qmodel.state_dict())
        
        print ('\r  It {} with reward {:.4f}'.format(it_episode, acc_reward), end='\n')
        if greater_reward < acc_reward:
            greater_reward = acc_reward
            Qmodel.save(os.path.join('pts', 'dql_model.pt'))
            icm.save(os.path.join('pts', 'icm_model.pt'))

    losses_ = np.array(losses)
    np.save(os.path.join('out', 'dql_losses.npy'), losses_)
    del icm 
    # plt.figure(figsize=(8,6))
    # plt.plot(np.log(losses_[:,0]),label='Q loss')
    # plt.plot(np.log(losses_[:,1]),label='Forward loss')
    # plt.plot(np.log(losses_[:,2]),label='Inverse loss')
    # plt.legend()
    # plt.show()

    del opt
    del replay

    # best model
    Qmodel.load(os.path.join('pts', 'dql_model.pt'))
    Qmodel.eval()
    it_episode, acc_reward = 0, 0.
    init_time = time.time()

    print ('# Ending:','Deep Q Learning algorithm')
    state1 = prepareBackpackState(*env.reset()).unsqueeze(0)
    with torch.no_grad():
        while True:
            # parafernalia ----------------------------
            it_episode += 1
            # -----------------------------------------
            q_val_pred = Qmodel(state1)
            action = int(__policy_dql(q_val_pred, nactions=BACKPACK_SIZE+1, eps=0.01))

            back_state, vec_state , e_reward, done = env.step(action)
            state1 = prepareBackpackState(back_state, vec_state).unsqueeze(0)
            acc_reward += e_reward
            
            all_obj_seeit = done
            if done:
                print ('\r  It {} with reward {:.4f} dt:{:.1f}'.format(it_episode, acc_reward, time.time()-init_time))
                break
            print ('\r  It {} with reward {:.4f} dt:{:.1f}'.format(it_episode, acc_reward, time.time()-init_time), end=' ')

    # esporting final state of the backpack
    env.export_prototypes(file_list  = [os.path.join('data','pos_center.txt'), os.path.join('data','neg_center.txt')], 
                          label_list = [1                                    , 0])
    del env

def extractPrototypes(method, params):
    """ Apply a method to extract prototypes from data. \n

    method: the method used to select prototypes, most be in [\'dql\', \'dql-intrinsic\']\n

    data_path:str a path to a \'.csv\' file with at most the columns [vcolumn, lcolumn]. \n
    eval_data_path: same as data_path, but treated as evaluation data \n
    
    The column vcolumn most be a list a floating points, a vector.\n

    The column lcolumn is the label of the vectors, [0,1]. """
    __paramu = {'intrinsic':True, 'lambda':0.1, 'eta':1.0, 'gamma':0.2, 'eps':0.15, 'beta':0.2,
                'lcolumn':'is_humor', 'vcolumn':'vecs', 'max_prototypes':20, # 200
                'batch_size':10, 'lr':0.001, 'epochs':20, 'memory_size':50}
    __paramu.update(params)

    methods_ = [('dql', __prototypes_with_dql), ('dql-intrinsic', __prototypes_with_dql)]

    for mname, fun in methods_:
        if method == mname:
            fun(__paramu)
            return
    print ('ERROR::extractPrototypes Method parameter', '\''+method+'\'', 'is not in [', ' , '.join(['\''+s+'\'' for s,_,_ in methods_]), '] !!!!')