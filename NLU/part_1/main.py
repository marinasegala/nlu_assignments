# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import *
from model import *
from utils import *
from torch.utils.data import DataLoader 
import torch.optim as optim

from tqdm import tqdm

import copy

if __name__ == "__main__":

    tmp_train_raw = load_data(os.path.join('part_1/dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('part_1/dataset','ATIS','test.json'))

    train_raw, dev_raw = create_dev_set(tmp_train_raw) #divide the tmp_train_raw for having train_raw and dev_raw

    lang = create_lang(train_raw, dev_raw, test_raw) 

    train_dataset = IntentsAndSlots(train_raw, lang) 
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    #----------------- SINGLE RUN ---------------------#
    '''
    #----------- Parameters settings ------------#
    hid_size = 200
    emb_size = 300

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    #------------# 
    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    best_model = None

    #----------- TRAINING AND EVAL ------------#
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        if x % 5 == 0: # check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break 

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    #----------- SAVING ------------#
    model_name = 'model_LSTM_21'
    path = os.path.join("bin", model_name)
    saving_object = {"epoch": x, 
                     "model": model.state_dict(), 
                     "optimizer": optimizer.state_dict(), 
                     "w2id": lang.word2id, 
                     "slot2id": lang.slot2id, 
                     "intent2id": lang.intent2id}
    torch.save(saving_object, path)

    path_info = 'PART_11/'
    save_infos (path_info, model_name, lr, hid_size, emb_size, losses_train, losses_dev, sampled_epochs, x, results_test, best_f1, intent_test, lang.word2id, lang.slot2id, lang.intent2id)
    
    '''

    #----------------- MULTIPLE RUNS  -----------------#
    #----------- Parameters settings ------------#
    hid_size = 200
    emb_size = 300

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    #------------#
    
    n_epochs = 200
    runs = 5
    best_model = None
    best_model_array = []  
    best_f1_array = []
    slot_f1s, intent_acc = [], []

    #----------- TRAINING AND EVAL ------------#

    for i in tqdm(range(0, runs)):
        # for each run, the model has to be reinitialized, as well as the optimizer, the criterions, the patience, the losses, the sampled epochs and the best f1
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                        vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0

        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            
            if x % 5 == 0: #for saving computational power, the evaluation is done every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                # implementation choices - saving the model with the best f1 score (the accuracy is not considered in this case)
                if f1 > best_f1: #if the f1_score is better than the previous one, the model is saved 
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to('cpu')
                    best_model_array.append(best_model)
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break 

        best_f1_array.append(best_f1)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
    
    #----------- SAVING ------------#
    model_name = 'model_NLU'
    path = 'part_1/bin/' + model_name + '.pt'

    index_best = np.argmax(best_f1_array)

    saving_object = {"epoch": x, 
                     "model": best_model_array[index_best].state_dict(), 
                     "optimizer": optimizer.state_dict(), 
                     "w2id": lang.word2id, 
                     "slot2id": lang.slot2id, 
                     "intent2id": lang.intent2id}
    torch.save(saving_object, path)