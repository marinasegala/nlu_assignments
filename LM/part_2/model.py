import torch 
import torch.nn as nn    

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        self.output.weight = self.embedding.weight #tie weights

    def forward(self, input_sequence):  #how layers interact
        emb = self.embedding(input_sequence)
        lsmt_out, _  = self.lstm(emb) #emb
        output = self.output(lsmt_out).permute(0,2,1) #rnn_out
        return output
        #implement variational dropout


# class VariationalDropout(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, dropout):
#         if self.training:
#             mask = torch.empty(input.size(1), input.size(2), device=input.device).bernoulli_(1 - dropout) / (
#                         1 - dropout)
#             mask = mask.expand_as(input)
#             return mask * input
#         else:
#             return input
#
#     def __repr__(self):
#         return "VariationalDropout()"