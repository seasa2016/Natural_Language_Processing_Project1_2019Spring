import torch
import torch.nn as nn
import torch.nn.functional as f

from .base import Base
from .Attention import Luong,Bahdanau

import math

class qalstm(Base):
    def __init__(self,args):
        super(qalstm,self).__init__(args)

        self.embeds_dim = args.embeds_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.batch_first = args.batch_first
        
        self.rnn = nn.LSTM(self.embeds_dim, self.hidden_dim, batch_first=self.batch_first , bidirectional=True, num_layers=self.num_layer)

        if(args.attention == 'luong'):
            self.attention = Luong(args.hidden_dim,args.hidden_dim)

        elif(args.attention == 'bahdanau'):
            self.attention = Bahdanau(8*args.hidden_dim,args.hidden_dim)
        else:
            raise ValueError('no this attention')

    def forward(self,querys,lengths,label=None):
        def pack(seq,seq_length):
            sorted_seq_lengths, indices = torch.sort(seq_length, descending=True)
            _, desorted_indices = torch.sort(indices, descending=False)
            if self.batch_first:
                seq = seq[indices]
            else:
                seq = seq[:, indices]
            packed_inputs = nn.utils.rnn.pack_padded_sequence(seq,
                                                            sorted_seq_lengths.cpu().numpy(),
                                                            batch_first=self.batch_first)
            return packed_inputs,desorted_indices

        def unpack(res, state,desorted_indices):
            padded_res,_ = nn.utils.rnn.pad_packed_sequence(res, batch_first=self.batch_first)
            state = [ _ for _ in state]

            for i in range(len(state)):
                state[i] = state[i][:,desorted_indices]
            
            if(self.batch_first):
                desorted_res = padded_res[desorted_indices]
            else:
                desorted_res = padded_res[:, desorted_indices]

            return desorted_res,state
            
        
        query_embs = [self.word_emb(querys[0]),self.word_emb(querys[1])]
        masks = [querys[0].eq(0),querys[1].eq(0)]

        query_result = []
        for query_emb,length,mask in zip(query_embs,lengths,masks):
            packed_inputs,desorted_indices = pack(query_emb,length)
            res, state = self.rnn(packed_inputs)
            query_res,_ = unpack(res, state,desorted_indices)
            query_result.append(query_res)
        
        """
        Attention part
        """
        att_results = self.attention(query_result,lengths,masks)
        
        """
        Aggregate
        """
        agg_results = [
            att_results[0].sum(dim=1).div(lengths[0].float().view(-1,1)),
            att_results[1].sum(dim=1).div(lengths[1].float().view(-1,1))
        ]


        result = torch.cat([agg_results[0],agg_results[1]],dim=1)


        out = self.linear(result,label=label)

        return out
