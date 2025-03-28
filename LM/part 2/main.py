# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    train_raw = read_file("part_2/dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("part_2/dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("part_2/dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))


    #----------- Parameters settings ------------#

    hid_size = 500 #200
    emb_size = 500 #300

    lr = 5 #0.0001 
    clip = 5 # Clip the gradient

    vocab_len = len(lang.word2id)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    #------------# 

    n_epochs = 100
    patience = 3 #prevent overfitting and save computational power
    losses_train = []
    losses_dev = []
    ppl_train_array = []
    ppl_dev_array = []
    sampled_epochs = []
    cut_epochs = []
    pbar = tqdm(range(1,n_epochs))

    best_ppl = math.inf
    best_model = None
    
    weights_update = {}
    best_weights = {}
    switch_optimizer = False

    stored_loss = math.inf
    hyp_control_monotonic = 2
    counting_weight = 0

    #----------- TRAINING AND EVAL ------------#

    for epoch in pbar:
        ppl, loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_train_array.append(ppl)
        losses_train.append(np.asarray(loss).mean())
        sampled_epochs.append(epoch)

        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        ppl_dev_array.append(ppl_dev)
        losses_dev.append(np.asarray(loss_dev).mean())

        pbar.set_description(f"PPL: {ppl_dev}, {switch_optimizer}")

        if  ppl_dev < best_ppl: #goal is to obtain a lower perplexity 
            best_ppl = ppl_dev 
            best_model = copy.deepcopy(model).to('cpu')

            for parameter in model.parameters(): #saving the parameter of the best model to use them as a restarting point
                best_weights[parameter] = parameter.data.clone() 

            patience = 3

        elif ppl_dev > best_ppl and switch_optimizer: #if the model is not improving but the optimazer is switched, the patience has to decrease
            patience -= 1
            
        if patience <= 0 and switch_optimizer: # Early stopping with patience
            break 

        
        #control if is the case so switch the optimizer (SGD to NonMonotonicAvSGD)
        if switch_optimizer == False and (len(losses_dev) > hyp_control_monotonic and  loss_dev > min(losses_dev[:-hyp_control_monotonic])):
            switch_optimizer = True 
            weights_update = best_weights #initialize the weights update with the best model
        
        if switch_optimizer: #if the optimizer is switched, weights are updated
            counting_weight += 1
            tmp = {}
            for parameter in model.parameters():
                tmp[parameter] = parameter.data.clone()
                weights_update[parameter] += tmp[parameter]
                
                average = weights_update[parameter] / counting_weight
                parameter.data = average.data.clone()
            
                 

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    
    #----------- SAVING ------------#
    
    name = 'model_LSTM'
    path = 'part_2/bin/' + name + '.pt'
    torch.save(model.state_dict(), path)

    path_info = 'part_2/'
    save_infos (path_info, name, lr, hid_size, emb_size, losses_train, losses_dev, ppl_train_array, ppl_dev_array, sampled_epochs, final_ppl, False)

