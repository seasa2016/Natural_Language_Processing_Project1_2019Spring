import torch
import torch.nn as nn


w = [1.0/16,1.0/15,1.0/5]
def count(pred,label)
    total = {}
    
    total['num'] = pred.shape[0]
    total['correct'] = (pred == label).sum()
    sample_weight = [w[i] for i in pred]
    total['weighted'] = metrics.accuracy_score(label.tolist(), pred.tolist(), normalize=True, sample_weight=sample_weight)

    return total

class Linear_two_class(nn.Module):
    def __init__(self,args):
        super(Linear_two_class,self).__init__()

        self.linear1 = nn.Linear(4*self.hidden_dim,self.hidden_dim)
        self.linear2_1 = nn.Linear(self.hidden_dim,2)
        self.linear2_2 = nn.Linear(self.hidden_dim,2)

        self.criterion = nn.KLDivLoss()
        self.w = [1.0/16,1.0/15,1.0/5]
    def forward(self,x,label=None):
        out = self.linear1(x)
        
        out_1 = self.linear2_1(F.relu(out))
        out_2 = self.linear2_2(F.relu(out))
        
        pred = (out_1.topk(1)[1]*(1+out_2.topk(1)[1])).view(-1)
        if(label is None):
            #return predict output
            return pred

        else:
            #return loss and acc
            total = {'loss':{},'count':{}}
            
            loss = self.criterion(F.log_softmax(out_1,dim=1),label[0]) 
            total['loss']['relation'] = loss.cpu().detach()
            total_loss = loss

            loss = self.criterion(F.log_softmax(out_2,dim=1),label[1]) 
            total['loss']['type'] += loss.cpu().detach()
            total_loss += loss
            
            total['count'] = count(pred,label[-1])
            
            return total_loss,total

class Linear_two_regression(nn.Module):
    def __init__(self,args):
        super(Linear_two_regression,self).__init__()

        self.linear1 = nn.Linear(4*self.hidden_dim,self.hidden_dim)
        self.linear2_1 = nn.Linear(self.hidden_dim,1)
        self.linear2_2 = nn.Linear(self.hidden_dim,1)

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.threshold1 = 0.5
        self.threshold2 = 0.5

    def forward(self,x,label=None):
        out = self.linear1(x)
        out_1 = self.linear2_1(F.relu(out))
        out_2 = self.linear2_2(F.relu(out))
                
        pred = ( (out_1>self.threshold1).int()*(1+(out_2>self.threshold2).int()) ).view(-1)
        if(label is None):
            #return predict output
            return pred
        else:
            #return loss and acc
            total = {'loss':{},'count':{}}
            
            loss = self.criterion(out_1,label[0]) 
            total['loss']['relation'] = loss.cpu().detach()
            total_loss = loss

            #drop out for unrelated data
            loss = label[0].float()*self.criterion(out_2,label[1]) 
            total['loss']['type'] += loss.cpu().detach()
            total_loss += loss
            
            total['count'] = count(pred,label[-1])
            
            return total_loss,total

class Linear_three_class(nn.Module):
    def __init__(self,args):
        super(Linear_three_class,self).__init__()

        self.linear1 = nn.Linear(4*self.hidden_dim,self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim,3)

        self.criterion = nn.CrossEntropyLoss()
    def forward(self,x,label=None):
        out = self.linear1(x)
        
        out = self.linear2(F.relu(out))
        
        pred = out.topk(1)[1].view(-1)
        if(label is None):
            #return predict output
            return pred
        else:
            #return loss and acc
            total = {'loss':{},'count':{}}
            
            loss = self.criterion(out,label[-1]) 
            total['loss']['total'] = loss.cpu().detach()
            total_loss = loss
            
            total['count'] = count(pred,label[-1])
            
            return total_loss,total

class Base(nn.Module):
    def __init__(self,args):
        super(Base, self).__init__()
        self.args = args

        self.word_emb =nn.Embedding(args.word_num,args.embeds_dim)

        if(args.mode == 'pretrain'):
            self.load()
            self.word_emb.weight.requires_grad = False
            print("here",self.word_emb.weight.requires_grad)

        if(args.pred=='linear_two_class'):
            self.linear = Linear_two_class(args)
        elif(args.pred=='linear_two_regression'):
            self.linear = Linear_two_regression(args)
        elif(args.pred=='linear_three_class'):
            self.linear = Linear_three_class(args)


    def load(self):
        if(self.args.embed_type == 'glove'):
            pass
        elif(self.args.embed_type == 'fasttext'):
            
            with open('./data/embedding/glove.6B.100d.txt') as f:
                arr = np.zeros((self.word_emb.weight.shape[0],self.word_emb.weight.shape[1]),dtype=np.float32)
                for i,line in enumerate(f):
                    for j,num in enumerate(line.strip().split()[1:]):
                        arr[i+1,j] = float(num)
                        
                self.word_emb.weight = nn.Parameter(torch.tensor(arr))
