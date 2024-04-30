# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertModel
from pprint import pprint

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results


    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    # print('Train samples:', len(tmp_train_raw))
    # print('Test samples:', len(test_raw))

    # pprint(tmp_train_raw[0])

    # Create the dev set
    train_raw, dev_raw = create_dev_set(tmp_train_raw, test_raw) #y_train, y_dev, y_test

    lang = create_lang(train_raw, dev_raw, test_raw) 

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model = BertModel.from_pretrained("bert-base-uncased") # Download the model

    inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
    pprint(inputs)