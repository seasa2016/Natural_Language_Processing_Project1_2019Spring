import torch
import torch.nn as nn
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

        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        return self.fc(sim)


