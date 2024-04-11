import torch 
import torch.nn as nn

class RNN_cell(nn.Module):
    def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()

        self.W = nn.Linear(input_size, hidden_size, bias=False) #manage input
        self.U = nn.Linear(hidden_size, hidden_size) # manage prevoius step h(t-1)
        self.V = nn.Linear(hidden_size, hidden_size) # maps the unit ??
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)
        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output
    

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        self.last_dropout = nn.Dropout(out_dropout)

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):  #how layers interact
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(drop1) #emb
        drop2 = self.last_dropout(rnn_out)
        output = self.output(drop2).permute(0,2,1) #rnn_out
        return output
    
