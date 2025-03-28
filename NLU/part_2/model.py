from collections import Counter
import torch

import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

class ModelIAS(BertPreTrainedModel):

    def __init__(self, config, hid_size, out_slot, out_int):
        super(ModelIAS, self).__init__(config)
        self.bert = BertModel(config=config)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int) 
       
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs_ids, attention_mask):

        outputs = self.bert(inputs_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pool_output = outputs.pooler_output

        utt_emb = self.dropout(last_hidden_state) #dropout layer
        slots = self.slot_out(utt_emb) # compute slot logits

        out = self.dropout(pool_output)
        intent = self.intent_out(out) # compute intent logits 
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1)
        # Slot size: batch_size, classes, seq_len
        return slots, intent