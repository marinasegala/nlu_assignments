import torch 
import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        self.last_dropout = nn.Dropout(out_dropout)

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):  #how layers interact
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        lsmt_out, _  = self.lstm(drop1)
        drop2 = self.last_dropout(lsmt_out)
        output = self.output(drop2).permute(0,2,1) 
        return output
    
