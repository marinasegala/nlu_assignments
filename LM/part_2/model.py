import torch 
import torch.nn as nn    

DEVICE = 'cuda:0'
import torch.distributions.bernoulli.Bernoulli as bern


class VarDropout(nn.Module):
    def __init__(self, drop = 0.5):
        super(VarDropout, self).__init__()
        self.drop = drop

    def forward(self, x):
        if not self.training:
            return x
        size_batch = x.size(0)
        size_emb = x.size(2)

        #create bernoulli mask  
        bernoulli_distribution = bern(1 - self.drop)

        m = bernoulli_distribution.sample(size_batch, 1, size_emb).to(DEVICE)  #x.new (size_batch, 1, size_emb).bernoulli_(1 - self.drop) #for each element in the batch, we will drop the same words
        mask = m / (1 - self.drop)
        mask = mask.expand_as(x)
        return mask * x
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VarDropout()
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.last_dropout = VarDropout()
        self.output.weight = self.embedding.weight #tie weights

    def forward(self, input_sequence):  #how layers interact
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(drop1) #emb
        drop2 = self.last_dropout(lstm_out)
        output = self.output(drop2).permute(0,2,1) #rnn_out
        return output
