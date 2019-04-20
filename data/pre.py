import numpy as np
import codecs
import pandas as pd

data = pd.read_csv('train.csv')

def convert(x):
    x['label'] = 'agreed'
    return x

data = data.apply(lambda x: x if(x['tid1'] != x['tid2']) else convert(x) ,axis=1)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
train.to_csv('local_train.csv')
test.to_csv('local_test.csv')