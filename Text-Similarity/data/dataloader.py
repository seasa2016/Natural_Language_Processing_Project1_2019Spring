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
	def __init__(self,file_name,mode='train',pred='two_class',maxlen=128,transform=True):
		self.mode = mode
		self.data = []
		self.pred = pred

		temp = pd.read_csv(file_name)
		if(mode=='test'):
			for i,query in enumerate(temp['query']):
				query = ast.literal_eval(query)
				query = [query[0][:maxlen],query[1][:maxlen]]
				length = [len(query[0]),len(query[1])]

				self.data.append({
					'query':query,
					'length':length
				})
				
		elif(mode=='train' or mode=='eval'):
			if('two_class' in pred):
				disagreed = [[.0,1.0],[.0,1.0],2]
				agreed = [[.0,1.0],[1.0,.0],1]
				unrelated = [[1.0,.0],[.0,.0],0]
			elif(pred=='two_regression'):
				disagreed = [[1],[1],2]
				agreed = [[1],[0],1]
				unrelated = [[0],[0],0]
			elif(pred=='three_class'):
				disagreed = [2]
				agreed = [1]
				unrelated = [0]
			else:
				raise ValueError('no this type {0}'.format(pred))

			for query,label in zip(temp['query'],temp['label']):
				query = ast.literal_eval(query)
				query = [query[0][:maxlen],query[1][:maxlen]]
				length = [len(query[0]),len(query[1])]

				if(length[0]==0 or length[1]==0):
					continue
				
				if(label=='disagreed'):
					l = disagreed
				elif(label=='agreed'):
					l =agreed
				elif(label=='unrelated'):
					l = unrelated
				
				self.data.append({
					'query':query,
					'length':length,
					'label':l
				})
				if(label=='agreed'):
					self.data.append({
						'query':query[::-1],
						'length':length[::-1],
						'label':l
					})

				
				
		self.transform = transform
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample = self.data[idx]

		out = {}
		if(transforms):
			for name in ['query','length']:
				out[name]=[]
				for i in range(len(sample[name])):
					out[name].append(torch.tensor(sample[name][i],dtype=torch.long))
			
			if('label' in sample):
				out['label']=[]
				if(self.pred=='two_class' or self.pred=='two_regression'):
					out['label'].append(torch.tensor(sample['label'][0],dtype=torch.float))
					out['label'].append(torch.tensor(sample['label'][1],dtype=torch.float))
				elif(self.pred=='three_class'):
					pass
				out['label'].append(torch.tensor(sample['label'][-1],dtype=torch.long))
		return out


def collate_fn(data):
	output = dict()

	output['length'] = [] 
	for i in range(len(data[0]['length'])):
		temp = [ _['length'][i] for _ in data]	 
		output['length'].append( torch.stack(temp, dim=0) )

	if('label' in data[0]):
		output['label'] = []
		for i in range(len(data[0]['label'])):
			temp = [ _['label'][i] for _ in data]	 
			output['label'].append( torch.stack(temp, dim=0) )



	#deal with source and target
	output['query'] = []
	for i in range(len(data[0]['query'])):
		length = output['length'][i]
		l = length.max().item()

		temp = []
		for j in range(len(data)):
			if(l-length[j].item()>0):
				temp.append( torch.cat([data[j]['query'][i],torch.zeros(l-length[j].item(),dtype=torch.long)],dim=-1) )
			else:
				temp.append( data[j]['query'][i] )
		
		output['query'].append( torch.stack(temp, dim=0).long() )

	return output

if(__name__ == '__main__'):
	data = itemDataset('./all_no_embedding/eval.csv',pred='linear_two_class')
	print(data[0])

	print(collate_fn([data[0],data[1]]))

