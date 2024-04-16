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
import csv

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
    hid_size = 200 #200
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

    optimizer = optim.SGD(model.parameters(), lr=lr) #optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    #-----------------------#

    n_epochs = 100
    patience = 3 #prevent overfitting and save computational power
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    array_ppl = []
    cut_epochs = []
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            array_ppl.append(ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                # if patience == 1:
                #   print(epoch, ' di 4')
                #   lr = lr/4
                # else:
                #   print(epoch, ' di 2')
                #print(epoch)
                cut_epochs.append(epoch)
                lr = lr /2


            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    print(cut_epochs)

    #save into a csv file the results
    
    index = 1
    with open('LM/part_1/results_'+str(index)+'.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Dev Loss', 'PPL'])
        for i in range(len(array_ppl)):
            writer.writerow([sampled_epochs[i], losses_train[i], losses_dev[i], array_ppl[i]])
    #-----------------------#

    # To save the model
    # path = 'model_bin/model_name.pt'
    # torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))

    #-----------------------#