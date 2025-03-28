# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import *
from model import *
from utils import *
from torch.utils.data import DataLoader 
import torch.optim as optim

from transformers import BertTokenizer, BertConfig

from tqdm import tqdm

import copy

if __name__ == "__main__":

    tmp_train_raw = load_data(os.path.join('part_2/dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('part_2/dataset','ATIS','test.json'))
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # download the tokenizer

    train_raw, dev_raw = create_dev_set(tmp_train_raw)  #divide the tmp_train_raw for having train_raw and dev_raw

    lang = create_lang(train_raw, dev_raw, test_raw, tokenizer) 

    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    #----------- Parameters settings ------------#
    
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    config = BertConfig.from_pretrained("bert-base-uncased")
    hid_size = config.hidden_size #768

    model = ModelIAS.from_pretrained("bert-base-uncased", config=config, hid_size = hid_size, out_slot = out_slot, out_int = out_int).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    #------------#

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = -1

    best_model = None
    
    #----------------- TRAINING + EVAL -----------------#
    
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                    criterion_intents, model, lang)
        losses_dev.append(np.asarray(loss_dev).mean())
        
        f1 = results_dev['total']['f']

        # implementation choices - saving the model with the best f1 score (the accuracy is not considered in this case)
        if f1 > best_f1:  #if the f1_score is better than the previous one, the model is saved 
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
    model_name = 'model_NLU'
    path = 'part_2/bin/' + model_name + '.pt'
    torch.save(model.state_dict(), path)

    path_info = 'part_2/'
    save_infos (path_info, model_name, lr, hid_size, losses_train, losses_dev, sampled_epochs, x, results_test['total']['f'], best_f1, intent_test['accuracy'])