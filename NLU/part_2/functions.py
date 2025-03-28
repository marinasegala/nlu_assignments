# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import random
import numpy as np

import torch
import torch.nn as nn

from conll import evaluate
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

def init_weights(mat): #initialize the weights only of the 2 linear layers inserted in the model (slot_out and intent_out)
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
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['inputs_ids'], sample['attention_mask'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)

            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['inputs_ids'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()

                utterance = [lang.tokenizer.convert_ids_to_tokens(elem) for elem in utt_ids] # ids to tokens 
               
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                #pos_pad = [index for index, value in enumerate(gt_ids) if value == 0]
                pos_pad = [] #extract the pad positions
                for index in range(len(gt_ids)):
                    if gt_ids[index] == 0:
                        pos_pad.append(index)

                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots) if elem != 'pad'])
                #only append the tokens that not correspond to the 'pad' gt_slot
                
                tmp_seq = [(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)]
                # tmp_seq = []
                # for id_el, elem in enumerate(to_decode):
                #     tmp_seq.append((utterance[id_el], lang.id2slot[elem]))

                hyp_slots.append([tmp_seq[id_el] for id_el, elem in enumerate(to_decode) if id_el not in pos_pad] )
                
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

def save_infos(path_info, model_name, lr, hid_size, losses_train, losses_dev, sampled_epochs, final_epoch, results_test, best_f1, intent_test):
    with open(path_info + model_name + '.txt', 'w') as f:
        f.write('Learning rate: ' + str(lr) + '\n')
        f.write('Hidden size: ' + str(hid_size) + '\n')
        #f.write('Final PPL: ' + str(final_ppl) + '\n')
        f.write('Last epoch: ' + str(final_epoch) + '\n')
        f.write('Slot F1: ' + str(results_test) + '\n')
        f.write('Best F1:' + str(best_f1) + '\n')
        f.write('Intent Accuracy:' + str(intent_test) + '\n')

    plt.plot(sampled_epochs, losses_dev, '-b', label='dev_loss')
    plt.plot(sampled_epochs, losses_train, '-r', label='train_loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(path_info+'loss.png')

    plt.clf()