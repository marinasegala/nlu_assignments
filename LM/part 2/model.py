import torch
import torch.nn as nn

DEVICE = 'cuda:0'
import torch.distributions.bernoulli as bernoulli

class VarDropout(nn.Module):
    def __init__(self, drop = 0.5):
        super(VarDropout, self).__init__()
        self.drop = drop

    def forward(self, x):
        if not self.training: #evaluation mode - no modification to x
            return x
        
        size_batch = x.size(0)
        size_emb = x.size(2)

        # create bernoulli mask with probability (1 - self.drop), used for applying dropout
        m = torch.empty(size_batch, 1, size_emb).bernoulli_(1 - self.drop).to(DEVICE)
        
        mask = m / (1 - self.drop) #normalization step
        mask = mask.expand_as(x) #to have the same size as x

        return mask * x #apply mask to x
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VarDropout() 
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.last_dropout = VarDropout()
        self.output = nn.Linear(hidden_size, output_size)
        
        self.output.weight = self.embedding.weight #tie weights 

    def forward(self, input_sequence):  # how layers interact
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.last_dropout(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output
