import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Base

class siamese(Base):
    def __init__(self, args):
        super(siamese, self).__init__(args)
        
        self.embeds_dim = args.embeds_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.batch_first = args.batch_first
        
        self.ln_embeds = nn.LayerNorm(args.embeds_dim)
        self.rnn = nn.LSTM(self.embeds_dim, self.hidden_dim, batch_first=self.batch_first , bidirectional=True, num_layers=self.num_layer)


        self.linear1 = nn.Linear(4*self.hidden_dim,self.hidden_dim)
        self.linear2_1 = nn.Linear(self.hidden_dim,2)
        self.linear2_2 = nn.Linear(self.hidden_dim,2)

    def forward(self, querys,lengths):
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

            state = [state[i][:,desorted_indices] for i in range(len(state)) ] 

            if(self.batch_first):
                desorted_res = padded_res[desorted_indices]
            else:
                desorted_res = padded_res[:, desorted_indices]

            return desorted_res,state

        def feat_extract(output,length,mask):
            """
            answer_output: batch*sentence*feat_len
            query_output:  batch*sentence*feat_len
            for simple rnn, we just take the output from 
            """
            if( self.batch_first == False ):
                output = output.transpose(0,1) 

            output = [torch.cat([ output[i][ length[i]-1 ][:self.hidden_dim] , 
                                        output[i][0][self.hidden_dim:]] , dim=-1 ) for i in range(length.shape[0])]
            output = torch.stack(output,dim=0)

            return output

        query_embs = [self.word_emb(querys[0]),self.word_emb(querys[1])]
        masks = [querys[0].eq(0),querys[1].eq(0)]

        query_result = []
        for query_emb,length,mask in zip(query_embs,lengths,masks):
            packed_inputs,desorted_indices = pack(query_emb,length)
            res, state = self.rnn(packed_inputs)
            query_res,_ = unpack(res, state,desorted_indices)
            query_result.append(feat_extract(query_res,length.int(),mask))
        
        query_result = torch.cat([query_result[0],query_result[1]],dim=1)
        
        out = self.linear1(query_result)
        
        out_1 = self.linear2_1(F.relu(out))
        out_2 = self.linear2_2(F.relu(out))
        return [out_1,out_2]


