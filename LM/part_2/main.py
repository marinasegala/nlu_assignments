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
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    #DEVICE = 'cuda:0'

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))


    #-----------------------#

    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    hid_size = 300 #200
    emb_size = 300

    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step

    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 2 #0.0001 # This is definitely not good for SGD
    clip = 5 # Clip the gradient
    device = 'cuda:0'

    vocab_len = len(lang.word2id)

    #model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr) #implement the new class NTASGD - non-monotonic adaptive SGD
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    #-----------------------#

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
    best_val_loss = []
    hyp_control_monotonic = 5
    counting_weight = 0

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        ppl, loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_train_array.append(ppl)
        losses_train.append(np.asarray(loss).mean())
        sampled_epochs.append(epoch)

        if epoch % 1 == 0:
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_array.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())

            pbar.set_description("PPL: %f" % ppl_dev)

            
            if  ppl_dev < best_ppl: 
                best_ppl = ppl_dev 
                best_model = copy.deepcopy(model).to('cpu')
                for parameter in model.parameters():
                    best_weights[parameter] = parameter.data.clone()
                    #saving the parameter of the best model for using them to restart in that point
                
                patience = 3
            elif ppl_dev > best_ppl and switch_optimizer: #if the model is not improving but the optimazer is switched, the patience is decreased
                patience -= 1
                lr = lr / 2

            if patience <= 0: # and switch_optimizer: # Early stopping with patience
                break # Not nice but it keeps the code clean

            #with this if we control if is the case so switch the optimizer (SGD to ASGD)
            if switch_optimizer == False and (len(losses_dev) > hyp_control_monotonic and losses_dev > min(best_val_loss[:-hyp_control_monotonic])):
                switch_optimizer = True 
                weights_update = best_weights 
            
            elif switch_optimizer:
                counting_weight += 1
                tmp = {}
                for parameter in model.parameters():
                    tmp[parameter] = parameter.data.clone()
                    weights_update[parameter] = tmp[parameter]
                    
                    average = weights_update[parameter] / counting_weight
                    parameter.data = average.data.clone()
                 

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    
    # To save the model
    name = 'model_LSTM_21'
    path = 'bin/' + name + '.pt'
    torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))

    #-----------------------#
    path_info = 'PART_21/'
    save_infos (path_info, name, lr, hid_size, emb_size, losses_train, losses_dev, ppl_train_array, ppl_dev_array, sampled_epochs, final_ppl, False)
    #-----------------------#
