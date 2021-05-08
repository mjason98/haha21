import os, sys, argparse
from code.models import setTransName, setSeed, makeDataSet, makeModels
from code.models import trainModels, evaluateModels, makeTrain_and_ValData
from code.models import convert2EncoderVec, predictWithPairModel
from code.models import makeDataSet_Prt, setFsize, setW
from code.utils  import projectData2D, makeParameters, parceParameter
from code.protos import extractPrototypes

# ================== PARAMETERS =============================

OUT_FOLDER       = 'out'
DATA_FOLDER      = 'data'
TRAIN_DATA_NAME  = 'data/haha_2018_train.csv'
EVAL_DATA_NAME   = 'a.csv'
TEST_DATA_NAME   = 'data/haha_2018_test_gold.csv'
TRAIN_ENCODER    = True
TRAIN_CENTERS    = True
TRAIN_CMP        = True

# ===========================================================

params = {
    'batch_encoder':128,
    'lr_encoder':5e-5,
    'epochs_encoder':15,
    'hsize_encoder':256,
    'dpr_encoder':0.0,
    'lr_factor_encoder':9/10,
    'optm_encoder':'adam', # rms
    'selop':'addn', #first
    'lcolumn':'is_humor', 
    'vcolumn':'vecs', 
    
    # Deep Reinforcement Learning Parameters ---------------
    'max_prototypes':52,
    'ICM':True, 
    'lambda':0.1, 
    'eta':1.0, 
    'gamma':0.9, 
    'eps':0.15, 
    'beta':0.2,
    'batch_size':64, 
    'lr':0.001, 
    'epochs':20, 
    'memory_size':100,
    
    'dropout':0.1,
    'nhead':3,
    'nhid':128,
    'd_model':90,
    'n_layers':3,
    'target_refill':200,
    'reduced_data_prototypes':True,
    'distribution_train':'85-5-5-5', #the sum of this most be 100
    # ---------------------------------------------------------

    # Siames Net parameters -----------------------------------
    'siam_batch':128, 
    'siam_lr':0.0001, 
    'siam_epochs': 50,
    'siam_hsize':64,
    'siam_dpr':0.05,
    #---GENERAL -----------------------------------------------
    'num_workers':4
}

def check_params(arg=None):
    global TEST_DATA_NAME
    global EVAL_DATA_NAME
    global TRAIN_DATA_NAME
    global TRAIN_ENCODER
    global TRAIN_CMP
    global TRAIN_CENTERS

    global params

    # INFOMAP_PATH = '/DATA/work_space/2-AI/3-SemEval21/infomap-master'
    # INFOMAP_EX   = 'Infomap'
    ONLINE_NAME = "/DATA/Mainstorage/Prog/NLP/dccuchile/bert-base-spanish-wwm-cased"
    # ONLINE_NAME  = "dccuchile/bert-base-spanish-wwm-cased"

    parse = argparse.ArgumentParser(description='Deep Model to solve IverLeaf2021 HAHA Task')

    parse.add_argument('-p', dest='predict', help='Unlabeled Data', 
                       required=False, default=TEST_DATA_NAME)
    parse.add_argument('-t', dest='train_data', help='Train Data', 
                       required=False, default=TRAIN_DATA_NAME)
    parse.add_argument('-d', dest='dev_data', help='Development Data', 
                       required=False, default=EVAL_DATA_NAME)

    parse.add_argument('--seed', dest='my_seed', help='Random Seed', 
                       required=False, default=123456)
    
    parse.add_argument('--trans_name', dest='trsn', help='The transformer name to pull from huggingface', 
                       required=False, default=ONLINE_NAME)
    parse.add_argument('--parameters', dest='params', help='file containing the parameters', 
                       required=False, default='')		

    parse.add_argument('--no_train_enc', help='Do not train the encoder model', 
					   required=False, action='store_false', default=True)
    parse.add_argument('--no_train_pro', help='Do not train the prototype model', 
					   required=False, action='store_false', default=True)
    parse.add_argument('--no_train_sia', help='Do not train the siames model', 
					   required=False, action='store_false', default=True)
    
    parse.add_argument('--make_parameter', help='Make a file with all parameters. Use this as reference.', 
					   required=False, action='store_true', default=False)
   
    returns = parse.parse_args(arg)

    TEST_DATA_NAME   = returns.predict
    TRAIN_DATA_NAME  = returns.train_data
    EVAL_DATA_NAME   = returns.dev_data

    
    ONLINE_NAME      = returns.trsn
    TRAIN_ENCODER    = bool(returns.no_train_enc)
    TRAIN_CMP        = bool(returns.no_train_sia)
    TRAIN_CENTERS    = bool(returns.no_train_pro)

    if bool(returns.make_parameter):
        makeParameters(params, os.path.join('parameters.txt'))
        return 0
    elif len(returns.params) > 0:
        if not os.path.isfile(returns.params):
            print ('#ERROR::parameter params \'{}\' is not a file, or not exist.'.format(returns.params))
            return 0
        else:
            __parm = parceParameter(returns.params)
            params.update(__parm)
    
    # Set Transformers staf
    setTransName(ONLINE_NAME)
    setFsize(int(params['d_model']))
    setW(int(params['num_workers']))

    # prepare environment
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.isdir(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)
    if not os.path.isdir('pts'):
        os.mkdir('pts')
    
    setSeed(int(returns.my_seed))
    return 1

def train_encoder():
    global TEST_DATA_NAME
    global EVAL_DATA_NAME
    global TRAIN_DATA_NAME

    t_data, t_loader = makeDataSet(TRAIN_DATA_NAME, batch=int(params['batch_encoder']), id_h='id', text_h='text', class_h='is_humor')
    e_data, e_loader = makeDataSet(EVAL_DATA_NAME,  batch=int(params['batch_encoder']), id_h='id', text_h='text', class_h='is_humor')

    model = makeModels('encoder', int(params['hsize_encoder']), dropout=float(params['dpr_encoder']), max_length=64, selection=str(params['selop']))
    trainModels(model, t_loader, epochs=int(params['epochs_encoder']), evalData_loader=e_loader, 
                nameu='encoder', optim=model.makeOptimizer(lr=float(params['lr_encoder']), lr_factor=float(params['lr_factor_encoder']), algorithm=str(params['optm_encoder'])))

    del t_loader
    del e_loader
    del t_data
    del e_data

    # Loading the best fit model
    model.load(os.path.join('pts', 'encoder.pt'))
    data, loader     = makeDataSet(TEST_DATA_NAME, batch=int(params['batch_encoder']), shuffle=False, id_h='id', text_h='text', class_h='is_humor')
    t_data, t_loader = makeDataSet(TRAIN_DATA_NAME,batch=int(params['batch_encoder']), shuffle=False, id_h='id', text_h='text', class_h='is_humor')
    e_data, e_loader = makeDataSet(EVAL_DATA_NAME, batch=int(params['batch_encoder']), shuffle=False, id_h='id', text_h='text', class_h='is_humor')

    # Make predictions using only the encoder
    evaluateModels(model, loader, name='pred_en', cleaner=[], header=('id', 'is_humor'))
    # Convert the data into vectors
    TRAIN_DATA_NAME = convert2EncoderVec('train_en', model, t_loader, save_as_numpy=True, df=DATA_FOLDER)
    EVAL_DATA_NAME  = convert2EncoderVec('dev_en',   model, e_loader, save_as_numpy=True, df=DATA_FOLDER)
    TEST_DATA_NAME  = convert2EncoderVec('test_en',  model, loader,   save_as_numpy=True, df=DATA_FOLDER)
    
    del t_loader
    del e_loader
    del t_data
    del e_data
    del loader
    del data 
    del model

def trainSiam():
    print ('# Start: Trianing Siamese Model')
    # temporal code, delete later ---------
    TRAIN_DATA_NAME = 'data/train_en.csv'
    EVAL_DATA_NAME  = 'data/dev_en.csv'
    TEST_DATA_NAME  = 'data/test_en.csv'
    # temporal code, delete later ---------

    # Siam data
    _, t_loader = makeDataSet_Prt(TRAIN_DATA_NAME, batch=params['siam_batch'], id_h='id', text_h='vecs', class_h='is_humor', criterion='random')
    _, e_loader = makeDataSet_Prt(EVAL_DATA_NAME, batch=params['siam_batch'], id_h='id', text_h='vecs', class_h='is_humor', criterion='random')

    model = makeModels('siam', int(params['siam_hsize']), int(params['d_model']), dropout=float(params['siam_dpr']))
    trainModels(model, t_loader, epochs=int(params['siam_epochs']), evalData_loader=e_loader,  
                nameu='siam', lr=float(params['siam_lr']), use_acc=False, b_fun=min)
    
    del t_loader
    del e_loader

    model.load(os.path.join('pts', 'siam.pt'))
    predictWithPairModel(TEST_DATA_NAME, model=model, out_name='pred_siam.csv')
    del model

if __name__ == '__main__':
    if check_params(arg=sys.argv[1:]) == 0:
        exit(0)

    if TRAIN_ENCODER:
        # This is temporal -------------------
        TRAIN_DATA_NAME, EVAL_DATA_NAME = makeTrain_and_ValData(TRAIN_DATA_NAME, class_label='is_humor', df=DATA_FOLDER)
        # This is temporal -------------------
        
        if not os.path.isfile(TEST_DATA_NAME):
            raise ValueError("File {} do not exist!".format(TEST_DATA_NAME))
        if not os.path.isfile(EVAL_DATA_NAME):
            raise ValueError("File {} do not exist!".format(EVAL_DATA_NAME))
        if not os.path.isfile(TRAIN_DATA_NAME):
            raise ValueError("File {} do not exist!".format(TRAIN_DATA_NAME))
        train_encoder()
        
    # temporal code, delete later ---------
    TRAIN_DATA_NAME = 'data/train_en.csv'
    EVAL_DATA_NAME  = 'data/dev_en.csv'
    TEST_DATA_NAME  = 'data/test_en.csv'
    # temporal code, delete later ---------
    if TRAIN_CENTERS:
        params.update({'data_path':TRAIN_DATA_NAME, 'eval_data_path':EVAL_DATA_NAME})
        extractPrototypes(method='dql', params=params)
        # projectData2D(os.path.join(DATA_FOLDER, 'train_en.csv'), use_centers=True, drops=['id', 'is_humor'])
        predictWithPairModel(TEST_DATA_NAME)
    
    if TRAIN_CMP:
        trainSiam()
