import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

import numpy as np
import json
import pandas as pd
import sys
import ast

class itemDataset(Dataset):
    def __init__(self,file_name,mode='train',transform=None):
        self.mode = mode
        self.data = []
        
        temp = pd.read_csv(file_name)
        if(mode=='test'):
            for query,length in zip(temp['query'],temp['length']):
                query = ast.literal_eval(query)
                length = ast.literal_eval(length)

                self.data.append({
                    'query':query,
                    'length':length
                })
                
        elif(mode=='train' or mode=='eval'):
            for query,length,label in zip(temp['query'],temp['length'],temp['label']):
                query = ast.literal_eval(query)
                length = ast.literal_eval(length)

                if(label=='disagreed'):
                    t1,t2 = [0.0,1.0],[0.0,1.0]
                    l = 0
                elif(label=='agreed'):
                    t1,t2 = [0.0,1.0],[1.0,0.0]
                    l = 1
                elif(label=='unrelated'):
                    t1,t2 = [1.0,0.0],[0.0,0.0]
                    l = 2

                self.data.append({
                    'query1':query[0],
                    'length1':length[0],
                    'query2':query[1],
                    'length2':length[1],
                    'label_relation':t1,
                    'label_type':t2,
                    'label':l
                })
                
                
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if(transforms):
            sample = self.transform(self.data[idx])
        return sample

class ToTensor(object):
    def __call__(self,sample):
        #print(sample)
        for name in ['query1','length1','query2','length2','label']:
            sample[name] = torch.tensor(sample[name],dtype=torch.long)

        for name in ['label_relation','label_type']:
            if(name in sample):
                sample[name] = torch.tensor(sample[name],dtype=torch.float)

        return sample

def collate_fn(data):
    output = dict()

    for name in ['length1','length2','label_relation','label_type','label']:
        temp = [ _[name] for _ in data]	 
        output[name] = torch.stack(temp, dim=0) 


    #deal with source and target
    for name in range(1,3):
        length = output['length{0}'.format(name)]
        name = 'query{0}'.format(name)
        l = length.max().item()

        for i in range(len(data)):
            if(l-length[i].item()>0):
                data[i][name] =  torch.cat([data[i][name],torch.zeros(l-length[i].item(),dtype=torch.long)],dim=-1)

        temp = [ _[name] for _ in data]
        output[name] = torch.stack(temp, dim=0).long()

    return {
        'length':[output['length1'],output['length2']],
        'query':[output['query1'],output['query2']],
        'label_relation':output['label_relation'],
        'label_type':output['label_type'],
        'label':output['label']
    }