from collections import Counter
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

max_seq_len = 50
PAD_TOKEN = 0
class Lang():
    def __init__(self, words, intents, slots, tokenizer):
        self.tokenizer = tokenizer 
        self.word2id = self.w2id(words)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements):
        return {word: self.tokenizer.convert_tokens_to_ids(word) for word in elements}
        # vocab = {'pad': PAD_TOKEN}
        # if unk:
        #     vocab['unk'] = len(vocab)
        # count = Counter(elements)
        # for k, v in count.items():
        #     if v > cutoff:
        #         vocab[k] = len(vocab)
        # return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
#customize dataset class
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.input_ids = []
        self.attention_mask = []
        
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.intents.append(x['intent'])
            #self.slots.append(x['slots'])

            tokens = tokenizer(x['utterance'])
            ids = tokenizer.convert_tokens_to_ids(tokens)
            slots = x['slots']
            
            for token, id, slot in zip(tokens, ids, slots):
                #print(f'Token: {token}, ID: {id}')
                if token.startswith('##'):
                    self.slots.append('pad')
                else:
                    self.slots.append(slot)

                if id == 101 or id == 102:
                    self.slots.append('pad')

                self.input_ids.append(id)
            
            

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        self.attention_mask = [1] * len(self.input_ids)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]

        inputs_ids = torch.Tensor(self.input_ids[idx])
        attentions_mask = torch.Tensor(self.attention_mask[idx])

        sample = {'inputs_ids': inputs_ids,  'attentions_mask': attentions_mask, 'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        if 'unk' not in mapper:
            mapper['unk'] = 0 
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
    def mapping_in(self, data, mapper):
        return [[mapper[token] if token in mapper else mapper[self.unk] for token in seq] for seq in data]
    
class ModelIAS(nn.Module):

    def __init__(self, model_type, hid_size, out_slot, out_int):
        super(ModelIAS, self).__init__()
        
        self.bert = model_type
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int) 
       
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs_ids, attention_mask):

        outputs = self.bert(inputs_ids, attention_mask)
        last_hidden_state = outputs[0]
        pool_output = outputs[1]

        utt_emb = self.dropout(last_hidden_state) #dropout layer
        slots = self.slot_out(utt_emb) # Compute slot logits

        intent = self.intent_out(pool_output) # Compute intent logits 
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent