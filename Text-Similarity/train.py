from time import gmtime, strftime
import os
import argparse

from data.dataloader import itemDataset,collate_fn,ToTensor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils


def convert(data,device):
	for name in data:
		data[name] = data[name].to(device)
	return data

def process(args):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	dataloader = get_data(args.batch_size)
	
    print("setting model")
    model = model.to(device=device)

	print(model)
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss(reduction='sum')
	
	loss_best = 100000000
	print("start training")
    model.zero_grad()
	for now in range(args.epoch):
		model.train()
        train(model,dataloader['train'],criterion,args.step)
        model.eval()
        eval(model,dataloader['eval'],criterion,loss_best)

def train(model,data,criterion,step):
    total={'loss':0,'count':0,'num':0}
    for i,data in enumerate(data):
        data = convert(data,device)

        #deal with the classfication part
        out = model(data['query'],data['length'])

        loss = criterion(out,data['query_label']) 
        loss.backward()

        total['count'] += (out.topk(1)[1] == data['query_label']).sum()
        total['loss'] += loss.cpu().detach()
        total['num'] += out.shape[0]
        
        if(i%1==0):
            optimizer.step()
            model.zero_grad()

        if(i%160==0):
            print(i,' testing loss(class):{0} acc:{3}/{4}'.format(total['loss'],total['count'],total['num']))
            total={'loss':0,'count':0,'num':0}
    

def eval(model,data,criterion,loss_best):
    total={'loss':0,'count':0,'num':0}
    for i,data in enumerate(data):
        with torch.no_grad():
            #first convert the data into cuda
            data = convert(data,device)

            #deal with the classfication part
            out = model(data['query'],data['length'])
            
            total['count'] += (out.topk(1)[1] == data['query_label']).sum()
            total['loss'] += criterion(out,data['query_label'].view(-1)) 
            total['num'] += out.shape[0]
            
    if(now%args.print_freq==0):
        print('*'*10)
        print(i,' testing loss(class):{0} acc:{3}/{4}'.format(total['loss'],total['count'],total['num']))
    
    check = {
            'args':args,
            'model':model.state_dict()
            }
    torch.save(check, './saved_models/{0}/step_{1}.pkl'.format(args.save,now))

    if(Loss['loss']<loss_best):
        torch.save(check, './saved_models/{0}/best.pkl'.format(args.save))
        loss_best = Loss['class']
        
def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', default=512, type=int)
	parser.add_argument('--dropout', default=0, type=float)
	parser.add_argument('--epoch', default=200, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	
	parser.add_argument('--word_dim', default=128, type=int)
	parser.add_argument('--hidden_dim', default=128, type=int)
	parser.add_argument('--num_layer', default=2, type=int)

	parser.add_argument('--learning_rate', default=0.005, type=float)
	parser.add_argument('--model',required=True, type=str)

	parser.add_argument('--print_freq', default=1, type=int)

	parser.add_argument('--save', required=True , type=str)
	
	args = parser.parse_args()

	#setattr(args, 'input_size', 4096)
	setattr(args, 'input_size', 49527)
	setattr(args,'batch_first',True)
	setattr(args,'use_char_emb',False)
	setattr(args, 'class_size',1)
	setattr(args, 'cate',32)

	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')

	if not os.path.exists('./saved_models/{0}'.format(args.save)):
		os.makedirs('./saved_models/{0}'.format(args.save))

	print('training start!')
	process(args)
	print('training finished!')
	


if(__name__ == '__main__'):
    main()