from collections import Counter
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

from transformers import BertTokenizer, BertModel

PAD_TOKEN = 0
'''
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
'''
class Lang_Bert():
    def __init__(self, words, intents, slots, tokenizer, cutoff=0):
        self.tokenizer = tokenizer #use the tokenizer of bert
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True) # self.tokenizer.convert_tokens_to_ids(words)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab
    
    
'''
#customize dataset class
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
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
'''

class IntentsAndSlots_Bert (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.input_ids = []
        self.attention_mask = []
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

            utt_bert = self.tokenizer(x['utterance'], padding = False)['input_ids']

            list_token = self.tokenizer.convert_tokens_to_ids(utt_bert)
            print(utt_bert)
            print(list_token)
            print(self.tokenizer.convert_ids_to_tokens(list_token))
            # for i in range(len(list_token)):
            #     if list_token[i] == '[CLS]' or list_token[i] == '[SEP]':
            #         self.slots.append('pad')
            #     # elif '##' in utt_bert[i]:
            #     #     self.slots.append('pad')
            #     else:
            #         if i < len(x['slots']):
            #             self.slots.append(x['slots'][i])
            #         self.slots.append(x['slots'][i])
            break

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
        self.inputs_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.attention_mask = [1] * len(self.inputs_ids)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        inputs_ids = torch.Tensor(self.inputs_ids[idx])
        attention_mask = torch.Tensor(self.attention_mask[idx])
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]

        print(len(self.slot_ids[idx]), len(self.inputs_ids[idx]))
        #  inputs = self.tokenizer.encode_plus(utterance, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True)

        
        sample = {'inputs_ids': inputs_ids, 'attention_mask': attention_mask , 'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper, utt = False): # Map sequences to number
        if 'unk' not in mapper:
            mapper['unk'] = 0 
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    token = self.tokenizer.tokenize(x)
                    tmp_seq.extend([mapper[x]] + [0] * (len(token) - 1))
                    # tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # def __init__(self, dataset, lang):
    #     self.utterances = [x['utterance'] for x in dataset]
    #     self.intents = [lang.intent2id[x['intent']] for x in dataset]
    #     self.slots = [[lang.slot2id.get(s, lang.slot2id['pad']) for s in x['slots']] for x in dataset]
    #     self.tokenizer = lang.tokenizer

    # def __len__(self):
    #     return len(self.utterances)

    # def __getitem__(self, idx):
    #     utterance = self.utterances[idx]
    #     intent = self.intents[idx]
    #     slots = self.slots[idx]

    #     # Usa il tokenizer BERT per convertire la frase in ID di token
    #     inputs = self.tokenizer.encode_plus(utterance, add_special_tokens=True, return_tensors='pt')

    #     # Converti le liste in tensori
    #     intent = torch.tensor(intent)
    #     slots = torch.tensor(slots)

    #     sample = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'intent': intent, 'slot': slots}
    #     return sample
    
'''
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.Bert(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)    
        self.slot_out = nn.Linear(hid_size, out_slot) #hid_size * 2 if bidirectional
        self.intent_out = nn.Linear(hid_size, out_int) #hid_size * 2 if bidirectional
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]

        ## Concatena gli stati nascosti finali delle direzioni avanti e indietro
        # last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        utt_emb = self.dropout(utt_encoded) #dropout layer

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
'''

class ModelIAS_Bert(nn.Module):

    def __init__(self, model_bert, hid_size, out_slot, out_int):
        super(ModelIAS_Bert, self).__init__()
        
        self.bert = model_bert
        self.slot_out = nn.Linear(hid_size, out_slot) #oppure self.bert.config.hidden_size
        self.intent_out = nn.Linear(hid_size, out_int) #oppure self.bert.config.hidden_size
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs_ids, attention_mask = None): 
        # BERT gestisce automaticamente le sequenze con padding utilizzando l'attention mask

        # Process the batch
        outputs = self.bert(inputs_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pool_output = outputs[1]

        utt_emb = self.dropout(last_hidden_state) #dropout layer
        slots = self.slot_out(utt_emb) # Compute slot logits

        #TODO: also compute dropout for the pool_output??
        intent = self.intent_out(pool_output) # Compute intent logits 
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent