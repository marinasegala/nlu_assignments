# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import *
from model import *
from utils import *

from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import copy

if __name__ == "__main__":

    train_raw = read_file("part_1/dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("part_1/dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("part_1/dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))


    #--------- Parameters settings --------------#

    hid_size = 200 
    emb_size = 300

    lr = 0.001 
    clip = 5

    vocab_len = len(lang.word2id)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr = lr)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    #------------# 
    
    n_epochs = 100
    patience = 3 #prevent overfitting and save computational power
    
    losses_train = []
    losses_dev = []
    ppl_train_array = []
    ppl_dev_array = []
    last_epoch = 0

    best_ppl = math.inf
    best_model = None

    change_lr = False

    pbar = tqdm(range(1,n_epochs))

    #------------ TRAINING + EVAL -----------#

    for epoch in pbar:
        ppl, loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_train_array.append(ppl)
        losses_train.append(np.asarray(loss).mean())

        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        ppl_dev_array.append(ppl_dev)
        losses_dev.append(np.asarray(loss_dev).mean())

        pbar.set_description("PPL: %f" % ppl_dev)
        
        if  ppl_dev < best_ppl: #want to minimize the perplexity 
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1
            # if lr > 0.01:
            #     lr = lr / 2
            #     change_lr = True
            #     optimizer = optim.SGD(model.parameters(), lr=lr)


        if patience <= 0: # Early stopping with patience
            last_epoch = epoch
            break # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    

    #----------- SAVING ------------#
    name = 'model_LSTM'
    path = 'part_1/bin/' + name + '.pt'
    torch.save(model.state_dict(), path)

    path_info = 'part_1/'
    save_infos (path_info, name, lr, hid_size, emb_size, losses_train, losses_dev, ppl_train_array, ppl_dev_array, last_epoch, final_ppl, change_lr)
