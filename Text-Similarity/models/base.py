import torch
import torch.nn as nn

class Base(nn.Module):
    def __init__(self,args):
        super(Base, self).__init__()
        self.args = args

        self.word_emb =nn.Embedding(args.word_num,args._dim)

        if(args.mode == 'pretrain'):
            self.load()
            self.word_emb.weight.requires_grad = False
            print("here",self.word_emb.weight.requires_grad)

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