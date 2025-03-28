# Add functions or classes used for data loading and preprocessing
import os
import json
from pprint import pprint
import torch
from collections import Counter
import torch.utils.data as data
from sklearn.model_selection import train_test_split
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU

PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def create_dev_set(tmp_train_raw):
    # First we get the 10% of the training set, then we compute the percentage of these examples 

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw 

class Lang():
    def __init__(self, words, intents, slots, tokenizer):
        self.tokenizer = tokenizer

        self.word2id = self.w2id(words)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, unk = True, cutoff = 0):
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
            if elem != ' ':
                vocab[elem] = len(vocab)
        return vocab
    
def create_lang(train_raw, dev_raw, test_raw, tokenizer):
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff

    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, however this depends on the research purpose

    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, tokenizer)
    return lang

#customize dataset class
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.unk = unk
        self.utterances = []
        self.intents = []
        self.slot_id = []

        self.input_ids = []
        self.attention_mask = []
        
        for x in dataset:
            self.intents.append(x['intent'])

            utt = x['utterance'].replace("'",'') #better to remove the apostrophe - gives problems in the tokenization process
            tokens_utt = tokenizer(utt) #tokenization of the utterance
            ids = tokens_utt['input_ids'] #extract the ids from tokens_utt - seeing that is composed by a list of tokens, a list of ids and attention mask
            tokens_list = tokenizer.convert_ids_to_tokens(ids) 
            slots = x['slots'].split()
            rows_input = []
            rows_slot = []

            index_slot = 0

            # iterate over the tokens and the ids in order to insert in the list of slot_id the 0s (corrisponding to the pad token)
            # when the token produced by bert is a special ones or if the inizial word is splitted in more tokens (sub-word tokenization)
            for token, id in zip(tokens_list, ids): 
                if token.startswith('##') or token == '.':
                    rows_slot.append(lang.slot2id['pad']) 
                elif token == '[CLS]' or token == '[SEP]':
                    rows_slot.append(lang.slot2id['pad'])
                else:
                    slot = slots[index_slot]
                    rows_slot.append(lang.slot2id[slot])
                    index_slot += 1

                rows_input.append(id)
            
            self.slot_id.append(rows_slot)
            self.input_ids.append(rows_input)

        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
        self.attention_mask = [[1 if i != 0 else 0 for i in x] for x in self.input_ids]
        # the attention mask has to be filled with 1s, but if the corresponding token is a pad token, 0 is inserted

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        slots = torch.Tensor(self.slot_id[idx])
        intent = self.intent_ids[idx]

        input_ids = torch.Tensor(self.input_ids[idx])
        attention = torch.Tensor(self.attention_mask[idx])

        sample = {'inputs_ids': input_ids, 'attention_mask': attention, 'slots': slots, 'intent': intent}
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

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x['inputs_ids']), reverse=True)

    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_inpt, _ = merge(new_item['inputs_ids'])
    attention_mask, _ = merge(new_item['attention_mask'])
    y_slots, y_lengths = merge(new_item['slots'])
    intent = torch.LongTensor(new_item['intent'])
    
    src_inpt = src_inpt.to(device) # We load the Tensor on our selected device
    attention_mask = attention_mask.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["inputs_ids"] = src_inpt
    new_item["attention_mask"] = attention_mask
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item