import torch
import torch.nn as nn
<<<<<<< HEAD
from base import Base

class SiaGRU(Base):
    def __init__(self, args):
        super(SiaGRU, self).__init__(args)
        
        self.ln_embeds = nn.LayerNorm(self.embeds_dim)
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=self.num_layer)
        
    def forward(self, datas,data_len):
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

			output = [torch.cat([ output[i][ length[i]-1 ][:self.hidden_size] , 
										output[i][0][self.hidden_size:]] , dim=-1 ) for i in range(length.shape[0])]
			output = torch.stack(output,dim=0)

			return output

        words = [self.word_emb(datas[0]),self.word_emb(datas[1])]
        
		
        for 
		packed_inputs,desorted_indices = pack(word,data_len)
		res, state = self.rnn(packed_inputs)
		query_res,_ = unpack(res, state,desorted_indices)
		query_result = feat_extract(query_res,data_len.int(),mask)
=======


class SiaGRU(nn.Module):
    def __init__(self, args):
        super(SiaGRU, self).__init__()
        self.args = args
        self.embeds_dim = args.embeds_dim
        self.embeds = nn.Embedding(args.num_word, self.embeds_dim)
        self.ln_embeds = nn.LayerNorm(self.embeds_dim)
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True,num_layers=self.num_layer)
        
    def forward_once(self, x):
        output, hidden, cell = self.gru(x)
        return hidden.squeeze()

    def forward(self, *input):
        # sent1: batch_size * seq_len
        sent1 = input[0]
        sent2 = input[1]

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.ln_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.ln_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        encoding1 = self.forward_once(x1)
        encoding2 = self.forward_once(x2)
>>>>>>> 74b0d0c67bd0816035c44eeb6bd8042684c0daec

        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        return self.fc(sim)


