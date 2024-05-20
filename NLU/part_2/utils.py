# Add functions or classes used for data loading and preprocessing
import os
import json
from pprint import pprint
import torch

# device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
device = 'cpu'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
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
    src_utt, _ = merge(new_item['inputs_ids'])
    attention_mask, _ = merge(new_item['attention_mask'])
    y_slots, y_lengths = merge(new_item['slots'])
    intent = torch.LongTensor(new_item['intent'])
    
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    attention_mask = attention_mask.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["inputs_ids"] = src_utt
    new_item["attention_mask"] = attention_mask
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item