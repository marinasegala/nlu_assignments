'''
Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. 
You can refer to this paper to have a better understanding of how to implement this: https://arxiv.org/abs/1902.10909. 
In this, one of the challenges of this is to handle the sub-tokenization issue.
Note: The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.
The models that you can experiment with are BERT-base or BERT-large.
Intent classification: accuracy
Slot filling: F1 score with conll
Dataset to use: ATIS
'''
# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertModel
from pprint import pprint
import torch.optim as optim
from tqdm import tqdm

import copy

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model_Bert = BertModel.from_pretrained("bert-base-uncased") # Download the model

    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    # print('Train samples:', len(tmp_train_raw))
    # print('Test samples:', len(test_raw))

    # pprint(tmp_train_raw[0])

    # Create the dev set
    train_raw, dev_raw = create_dev_set(tmp_train_raw, test_raw) #y_train, y_dev, y_test

    lang = create_lang(train_raw, dev_raw, test_raw, tokenizer) 

    train_dataset = IntentsAndSlots_Bert(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots_Bert(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots_Bert(test_raw, lang, tokenizer)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
    # pprint(inputs)

    hid_size = 768
    emb_size = 300

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS_Bert(model_Bert, hid_size, out_slot, out_int).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    #----------------- TRAINING + EVAL -----------------#

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    best_model = None

    for x in tqdm(range(1,n_epochs)):
        #train_loader = preprocess(train_loader, tokenizer)
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    # PATH = os.path.join("bin", model_name)
    model_name = 'model_LSTM_21'
    path = os.path.join("bin", model_name)
    saving_object = {"epoch": x, 
                     "model": model.state_dict(), 
                     "optimizer": optimizer.state_dict(), 
                     "w2id": lang.word2id, 
                     "slot2id": lang.slot2id, 
                     "intent2id": lang.intent2id}
    torch.save(saving_object, path)

    path_info = 'PART_21/'
    save_infos (path_info, model_name, lr, hid_size, emb_size, losses_train, losses_dev, sampled_epochs, x, results_test, best_f1, intent_test, lang.slot2id, lang.intent2id)
    