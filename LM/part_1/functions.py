# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def save_infos(path, name, lr, hid_size, emb_size, losses_train, losses_dev, ppl_train_array, ppl_dev_array, sampled_epochs, final_ppl, bool):
    #all'interno della cartella creata, salva i parametri del modello e i risultati
    with open(path + name + '_dim.txt', 'w') as f:
        f.write('Learning rate: ' + str(lr) + '\n')
        f.write('Hidden size: ' + str(hid_size) + '\n')
        f.write('Embedding size: ' + str(emb_size) + '\n')
        f.write('Final PPL: ' + str(final_ppl) + '\n')
        f.write('Last epoch: ' + str(sampled_epochs[len(sampled_epochs)-1]))
        f.write('\nHalve : ' + str(bool))

    plt.plot(sampled_epochs, losses_dev, '-b', label='dev_loss')
    plt.plot(sampled_epochs, losses_train, '-r', label='train_loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(path+'loss.png')

    plt.clf()

    plt.plot(sampled_epochs, ppl_dev_array, '-b', label='dev')
    plt.plot(sampled_epochs, ppl_train_array, '-r', label='train')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(path+'ppl.png')
