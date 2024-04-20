import torch
import torch.nn as nn

DEVICE = 'cuda:0'
import torch.distributions.bernoulli.Bernoulli as bern


class VarDropout(nn.Module):
    def __init__(self, drop = 0.5):
        super(VarDropout, self).__init__()
        self.drop = drop

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        if self.mask is None:
            m = torch.empty(x.size(1), x.size(2), device = x.device).bernoulli_(1 - dropout)
            mask = m / (1 - dropout)
            mask = mask.expand_as(x)
            self.mask = mask
        return self.mask * x
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VarDropout() #nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.last_dropout = VarDropout() #nn.Dropout(out_dropout)
        
        self.output.weight = self.embedding.weight #tie weights

    def forward(self, input_sequence):  # how layers interact
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        lstm_out, _  = self.rnn(drop1) #emb
        drop2 = self.last_dropout(lstm_out)
        output = self.output(drop2).permute(0,2,1) #rnn_out
        return output
        #implement variational dropout
