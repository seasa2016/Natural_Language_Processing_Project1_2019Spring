from time import gmtime, strftime
import os
import argparse

from data.dataloader import itemDataset,collate_fn,ToTensor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

def get_data(train_file,eval_file,batch_size):
    train_dataset = itemDataset( file_name=train_file,mode='train',transform=transforms.Compose([ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=16,collate_fn=collate_fn)
    
    eval_dataset = itemDataset( file_name=eval_file,mode='eval',transform=transforms.Compose([ToTensor()]))
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,shuffle=True, num_workers=16,collate_fn=collate_fn)
    
    return {
        'train':train_dataloader,
        'eval':eval_dataloader
    }

def convert(data,device):
    for name in data:
        if(type(data[name])==list):
            for i in range(len(data[name])):
                data[name][i] = data[name][i].to(device)
        else:
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
	dataloader = get_data(os.path.join(args.data,'train.csv'),os.path.join(args.data,'eval.csv'),args.batch_size)

	print("setting model")
	if(args.model=='siamese'):
		model = siamese(args)
	model = model.to(device=device)

	print(model)
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
	criterion = nn.KLDivLoss()

	loss_best = 100000000
	print("start training")

    model.zero_grad()
	for now in range(args.epoch):
		model.train()
        train(model,dataloader['train'],criterion,args.step)
        model.eval()
        loss_best = eval(model,dataloader['eval'],criterion,loss_best)

def train(model,data,criterion,step):
    total={'loss_relation':0,'loss_type':0,'count':0,'num':0}
    for i,data in enumerate(data):
        data = convert(data,device)

        #deal with the classfication part
        out = model(data['query'],data['length'])
        
        loss = criterion(F.log_softmax(out[0],dim=1),data['label_relation']) 
        loss.backward(retain_graph=True)
        total['loss_relation'] += loss.cpu().detach()
        
        loss = criterion(F.log_softmax(out[1],dim=1),data['label_type']) 
        loss.backward(retain_graph=True)
        total['loss_type'] += loss.cpu().detach()
        total['num'] += out[0].shape[0]
        
        total['count'] += ((out[0].topk(1)[1]*(1+out[1].topk(1)[1])).view(-1) == data['label']).sum()
        
        if(i%1==0):
            optimizer.step()
            model.zero_grad()

        if(i%160==0):
            print(i,' train loss(relation):{0} loss(type):{1} acc:{2}/{3}'.format(total['loss_relation'],total['loss_type'],total['count'],total['num']))
            total={'loss_relation':0,'loss_type':0,'count':0,'num':0}

    

def eval(model,data,criterion,loss_best):
    total={'loss_relation':0,'loss_type':0,'count':0,'num':0}
    for i,data in enumerate(data):
        with torch.no_grad():
            #
            data = convert(data,device)
            out = model(data['query'],data['length'])

            loss = criterion(F.log_softmax(out[0],dim=1),data['label_relation']) 
            total['loss_relation'] += loss.cpu().detach()

            loss = criterion(F.log_softmax(out[1],dim=1),data['label_type'])
            total['loss_type'] += loss.cpu().detach()
            total['num'] += out[0].shape[0]

            total['count'] += out[0].topk(1)[1]*(1+out[1].topk(1)[1]) == data['label']
        
    print(i,' test loss(relation):{0} loss(type):{1} acc:{2}/{3}'.format(total['loss_relation'],total['loss_type'],total['count'],total['num']))
    
    check = {
            'args':args,
            'model':model.state_dict()
            }
    torch.save(check, './saved_models/{0}/step_{1}.pkl'.format(args.save,now))

    if(Loss['loss']<loss_best):
        torch.save(check, './saved_models/{0}/best.pkl'.format(args.save))
        loss_best = Loss['class']
    
    return loss_best
        
def main():
	parser = argparse.ArgumentParser()

	
	parser.add_argument('--.batch_size', default=32, type=int)
	parser.add_argument('--.dropout', default=0, type=int)
	parser.add_argument('--.gpu', default=0, type=int)
	parser.add_argument('--.embeds_dim', default=128, type=int)
	parser.add_argument('--.hidden_dim', default=128, type=int)
	parser.add_argument('--.num_layer', default=2, type=int)
	parser.add_argument('--.learning_rate', default=0.0001, type=float)
	
	parser.add_argument('--.print_freq', default=1, type=int)
	parser.add_argument('--.input_size', default=49527, type=int)
	parser.add_argument('--.batch_first', default=True, type=bool)
	parser.add_argument('--.data', default='./data/all_no_embedding/', type=str)
	parser.add_argument('--.mode' , default= 'train', type=str)
	parser.add_argument('--.step ', default= 1, type=int)
	parser.add_argument('--.model ', default= 'siamese', type=str)

	parser.add_argument('--.model', require=True)
	parser.add_argument('--.save', require=True)
	
	args = parser.parse_args()

	with open('{0}/vocab'.format(parser.add_argument('--.data)) as f:
		args.word_num = len(f.readlines())
	
	
	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')

	if not os.path.exists('./saved_models/{0}'.format(args.save)):
		os.makedirs('./saved_models/{0}'.format(args.save))

	print('training start!')
	process(args)
	print('training finished!')
	



if(__name__ == '__main__'):
    main()