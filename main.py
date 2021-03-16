import os, sys, argparse
from code.models import setTransName, setSeed, makeDataSet, makeModels
from code.models import trainModels, evaluateModels, makeTrain_and_ValData
from code.models import convert2EncoderVec

# ================== PARAMETERS =============================

OUT_FOLDER       = 'out'
DATA_FOLDER      = 'data'
TRAIN_DATA_NAME  = 'data/haha_2018_train.csv'
EVAL_DATA_NAME   = 'a.csv'
TEST_DATA_NAME   = 'data/haha_2018_test_gold.csv'
TRAIN_ENCODER    = True
TRAIN_CENTERS    = True
TRAIN_CMP        = True

BATCH_ENCODER    = 128
LR_ENCODER       = 5e-5
EPOCHS_ENCODER   = 15
HSIZE_ENCODER    = 256
DPR_ENCODER      = 0.0
LRFACTOR_ENCODER = 9/10
OPTM_ENCODER     = 'adam'
SELOP            = 'addn'

# ===========================================================

def check_params(arg=None):
    global BATCH_ENCODER
    global LR_ENCODER
    global EPOCHS_ENCODER
    global HSIZE_ENCODER
    global DPR_ENCODER
    global LRFACTOR_ENCODER
    global OPTM_ENCODER
    global SELOP
    
    global TEST_DATA_NAME
    global EVAL_DATA_NAME
    global TRAIN_DATA_NAME
    global TRAIN_ENCODER

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

    parse.add_argument('--lre', dest='lr_encod', help='The encoder learning rate to use in the optimizer', 
                       required=False, default=LR_ENCODER)
    parse.add_argument('--dpe', dest='dropout_encod', help='Dropout in the encoder', 
                       required=False, default=DPR_ENCODER)
    parse.add_argument('--se', dest='encod_size', help='The size of the dense layer in the encoder', 
                       required=False, default=HSIZE_ENCODER)
    parse.add_argument('--be', dest='batchs_encod', help='Number of batchs in the encoder training process', 
                       required=False, default=BATCH_ENCODER)
    parse.add_argument('--ee', dest='epochs_encod', help='Number of epochs in the encoder training process', 
                       required=False, default=EPOCHS_ENCODER)
    parse.add_argument('--lfe', dest='lfactor_encod', help='The learning rate factor to use in the encoder', 
                       required=False, default=LRFACTOR_ENCODER)

    parse.add_argument('--seed', dest='my_seed', help='Random Seed', 
                       required=False, default=123456)
    # parse.add_argument('--infomap-path', dest='ipath', help='Path to infomap executable', 
    #                    required=False, default=INFOMAP_PATH)
    # parse.add_argument('--infomap-name', dest='iname', help='Infomap executable name', 
    #                    required=False, default=INFOMAP_EX)
    
    parse.add_argument('--optim_e', dest='optim_encoder', help='Optimazer to use in the training process', 
                       required=False, default=OPTM_ENCODER, choices=['adam', 'rms'])
    # parse.add_argument('--etha', dest='etha', help='The multi task learning parameter to calculate the liner convex combination: \\math\\{loss = \\ethaL_1 + (1 - \\etha)L_2\\}', 
    #                    required=False, default=MTL_ETHA)		
    parse.add_argument('--trans-name', dest='trsn', help='The transformer name to pull from huggingface', 
                       required=False, default=ONLINE_NAME)		
    parse.add_argument('--vector-op', dest='selec', help='Operation over the last layer in the transformer to fit the last dense layer of the encoder', 
                       required=False, default=SELOP, choices=['addn', 'first'])
    parse.add_argument('--no_train_enc', help='Do not train the encoder', 
					   required=False, action='store_false', default=True)
   
    returns = parse.parse_args(arg)
    
    LR_ENCODER       = float(returns.lr_encod)
    DPR_ENCODER      = float(returns.dropout_encod)
    HSIZE_ENCODER    = int(returns.encod_size)
    BATCH_ENCODER    = int(returns.batchs_encod)
    EPOCHS_ENCODER   = int(returns.epochs_encod)
    LRFACTOR_ENCODER = float(returns.lfactor_encod)
    
    TEST_DATA_NAME   = returns.predict
    TRAIN_DATA_NAME  = returns.train_data
    EVAL_DATA_NAME   = returns.dev_data
    
    OPTM_ENCODER     = returns.optim_encoder
    SELOP            = returns.selec
    ONLINE_NAME      = returns.trsn
    TRAIN_ENCODER    = bool(returns.no_train_enc)
    
    # Set Infomap staf
    # INFOMAP_EX = returns.iname 
    # INFOMAP_PATH = returns.ipath
    # setInfomapData(INFOMAP_PATH, INFOMAP_EX)

    # Set Transformers staf
    setTransName(ONLINE_NAME)

    # prepare environment
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.isdir(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)
    if not os.path.isdir('pts'):
        os.mkdir('pts')
    
    setSeed(int(returns.my_seed))

    if not TRAIN_ENCODER and not os.path.isfile(os.path.join('pts', 'encoder.pt')):
        print('# Disabled the parameter \'--no_train_enc\'\n because there is not any checkpoint of the encoder under the name \'encoder.pt\' in folder \'pts\'')
        TRAIN_ENCODER = True

def train_encoder():
    global TEST_DATA_NAME
    global EVAL_DATA_NAME
    global TRAIN_DATA_NAME

    # This is temporal -------------------
    TRAIN_DATA_NAME, EVAL_DATA_NAME = makeTrain_and_ValData(TRAIN_DATA_NAME, class_label='is_humor', df=DATA_FOLDER)
    # This is temporal -------------------

    t_data, t_loader = makeDataSet(TRAIN_DATA_NAME, batch=BATCH_ENCODER, id_h='id', text_h='text', class_h='is_humor')
    e_data, e_loader = makeDataSet(EVAL_DATA_NAME,  batch=BATCH_ENCODER, id_h='id', text_h='text', class_h='is_humor')

    model = makeModels('encoder', HSIZE_ENCODER, dropout=DPR_ENCODER, max_length=64, selection=SELOP)
    trainModels(model, t_loader, epochs=EPOCHS_ENCODER, evalData_loader=e_loader, 
                nameu='encoder', optim=model.makeOptimizer(lr=LR_ENCODER, lr_factor=LRFACTOR_ENCODER, algorithm=OPTM_ENCODER))

    del t_loader
    del e_loader
    del t_data
    del e_data

    # Loading the best fit model
    model.load(os.path.join('pts', 'encoder.pt'))
    data, loader     = makeDataSet(TEST_DATA_NAME, batch=BATCH_ENCODER, shuffle=False, id_h='id', text_h='text', class_h='is_humor')
    t_data, t_loader = makeDataSet(TRAIN_DATA_NAME,batch=BATCH_ENCODER, shuffle=False, id_h='id', text_h='text', class_h='is_humor')
    e_data, e_loader = makeDataSet(EVAL_DATA_NAME, batch=BATCH_ENCODER, shuffle=False, id_h='id', text_h='text', class_h='is_humor')

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

if __name__ == '__main__':
    check_params(arg=sys.argv[1:])
    
    if TRAIN_ENCODER:
        train_encoder()