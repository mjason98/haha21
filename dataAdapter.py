import argparse, sys

from requests import head
from code.models import makeTrain_and_ValData
import pandas as pd

TEST_DATA_NAME   = ""
TRAIN_DATA_NAME  = ""
EVAL_DATA_NAME   = ""
DATA_FOLDER ="data"

def check_params(arg=None):
    global TEST_DATA_NAME
    global TRAIN_DATA_NAME

    parse = argparse.ArgumentParser(description='Deep Model to solve IverLeaf2021 HAHA Task')

    parse.add_argument('-p', dest='predict', help='Unlabeled Data', 
                       required=False, default=TEST_DATA_NAME)
    parse.add_argument('-t', dest='train_data', help='Train Data', 
                       required=False, default=TRAIN_DATA_NAME)
    
   
    returns = parse.parse_args(arg)

    TEST_DATA_NAME   = returns.predict
    TRAIN_DATA_NAME  = returns.train_data

    return 1

def transformData(dataPath, newName, hri=False):
    data = pd.read_csv(dataPath)
    
    ids = data['index'].map(lambda p: p.replace('tweet', ''))
    data.drop(['index'], axis=1, inplace=True)
    
    newColumns = ','.join(data.columns)
    newColumns = newColumns.replace('index', 'id').replace('encoding', 'vecs').replace('ground_humor', 'is_humor').split(',')
    newColumns = ['id'] + newColumns 
    
    data = pd.concat([ids, data], axis=1)
    del ids

    if hri:
        newColumns = newColumns + ['humor_rating']
        hr = pd.Series([0]*len(data))
        data = pd.concat([data, hr], axis=1)
        del hr 
    
    data.to_csv(newName, index=None, header=newColumns)

if __name__ == '__main__':
    if check_params(arg=sys.argv[1:]) == 0:
        exit(0)
    
    TRAIN_DATA_NAME, EVAL_DATA_NAME = makeTrain_and_ValData(TRAIN_DATA_NAME, class_label='ground_humor',
                                                            df=DATA_FOLDER, sentence_label='encoding')
    
    TRAIN_DATA_NAME = transformData(TRAIN_DATA_NAME, 'data/train_en.csv', hri=True)
    EVAL_DATA_NAME = transformData(EVAL_DATA_NAME, 'data/dev_en.csv', hri=True)
    TEST_DATA_NAME = transformData(TEST_DATA_NAME, 'data/test_en.csv', hri=False)