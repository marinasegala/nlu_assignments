# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from model import Lang
import torch
import torch.nn as nn

from conll import evaluate
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

def create_dev_set(tmp_train_raw, test_raw):
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

    y_test = [x['intent'] for x in test_raw]

    return train_raw, dev_raw #, y_train, y_dev, y_test
    '''
    # Intent distributions
    print('Train:')
    pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    print('Dev:'), 
    pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    print('Test:') 
    pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    print('='*89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))
    '''

def create_lang(train_raw, dev_raw, test_raw, tokenizer):
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff

    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, however this depends on the research purpose

    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, tokenizer)
    return lang

def init_weights(mat):
    for n, m in mat.named_modules():
        if n in ['slot_out', 'intent_out']:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['inputs_ids'], sample['attention_mask'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['inputs_ids'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def save_infos(path_info, model_name, lr, hid_size, emb_size, losses_train, losses_dev, sampled_epochs, final_epoch, results_test, best_f1, intent_test, word2id, slot2id, intent2id):
    #all'interno della cartella creata, salva i parametri del modello e i risultati
    with open(path_info + model_name + '.txt', 'w') as f:
        f.write('Learning rate: ' + str(lr) + '\n')
        f.write('Hidden size: ' + str(hid_size) + '\n')
        f.write('Embedding size: ' + str(emb_size) + '\n')
        #f.write('Final PPL: ' + str(final_ppl) + '\n')
        f.write('Last epoch: ' + str(final_epoch) + '\n')
        f.write('Slot F1: ', results_test['total']['f'], '\n')
        f.write('Best F1:', best_f1, '\n')
        f.write('Intent Accuracy:', intent_test['accuracy'], '\n')
        f.write('Computed vocabulary: \n\t vocab -' 
                + str((len(word2id)-2)) + '\n\t slots -' # we remove pad and unk from the count of word2id
                + str((len(slot2id))) + '\n\t intent -'
                + str((len(intent2id))) )

    plt.plot(sampled_epochs, losses_dev, '-b', label='dev_loss')
    plt.plot(sampled_epochs, losses_train, '-r', label='train_loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(path_info+'loss.png')

    plt.clf()