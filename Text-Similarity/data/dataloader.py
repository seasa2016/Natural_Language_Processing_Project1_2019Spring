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
	def __init__(self,file_name,mode='train',pred='linear_two_class',maxlen=128,transform=True):
		self.mode = mode
		self.data = []
		self.pred = pred

		temp = pd.read_csv(file_name)
		if(mode=='test'):
			for query in temp['query']:
				query = ast.literal_eval(query)
				query = [query[0][:maxlen],query[1][:maxlen]]
				length = [len(query[0]),len(query[1])]

				self.data.append({
					'query':query,
					'length':length
				})
				
		elif(mode=='train' or mode=='eval'):
			if(pred=='linear_two_class'):
				disagreed = [[.0,1.0],[.0,1.0],2]
				agreed = [[.0,1.0],[1.0,.0],1]
				unrelated = [[1.0,.0],[.0,.0],0]
			elif(pred=='linear_two_regression'):
				disagreed = [[1],[1],2]
				agreed = [[1],[0],1]
				unrelated = [[0],[0],0]
			elif(pred=='linear_three_class'):
				disagreed = [2]
				agreed = [1]
				unrelated = [0]

			for query,label in zip(temp['query'],temp['label']):
				query = ast.literal_eval(query)
				query = [query[0][:maxlen],query[1][:maxlen]]
				length = [len(query[0]),len(query[1])]

				if(label=='disagreed'):
					l = disagreed
				elif(label=='agreed'):
					l =agreed
				elif(label=='unrelated'):
					l = unrelated

				self.data.append({
					'query1':query[0],
					'length1':length[0],
					'query2':query[1],
					'length2':length[1],
					'label':l
				})
				
				
		self.transform = transform
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample = self.data[idx]

		out = {}
		if(transforms):
			for name in ['query1','length1','query2','length2']:
				out[name] = torch.tensor(sample[name],dtype=torch.long)

			out['label']=[]
			if(self.pred=='linear_two_class' or self.pred=='linear_two_regression'):
				out['label'].append(torch.tensor(sample['label'][0],dtype=torch.float))
				out['label'].append(torch.tensor(sample['label'][1],dtype=torch.float))
			elif(self.pred=='linear_three_class'):
				pass
			out['label'].append(torch.tensor(sample['label'][-1],dtype=torch.long))
		return out


def collate_fn(data):
	output = dict()

	for name in ['length1','length2']:
		temp = [ _[name] for _ in data]	 
		output[name] = torch.stack(temp, dim=0) 

	output['label'] = []
	for i in range(len(data[0]['label'])):
		temp = [ _['label'][i] for _ in data]	 
		output['label'].append( torch.stack(temp, dim=0) )



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
		'label':output['label']
	}

if(__name__ == '__main__'):
	data = itemDataset('./all_no_embedding/eval.csv',pred='linear_two_class')
	print(data[0])

	print(collate_fn([data[0],data[1]]))

