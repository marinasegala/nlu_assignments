import csv
import matplotlib.pyplot as plt
import numpy as np

def create_array(index):
    array = []
    count = 0
    inputfile = csv.reader(open('result_0.csv','r'))
    for row in inputfile:  
        if row[index] != '' and count != 0:
            array.append(int(float(row[index])))
        count += 1
    return array



epochs = create_array(0)
train_loss = create_array(1)
dev_loss = create_array(2)
pll_train = create_array(3)
pll_dev = create_array(4)

#Epoch,Train Loss,Dev Loss,PPL train,PPL dev
#plt.plot(epochs, pll, '-b.', markevery=markers_res_0, label='pll')
plt.plot(epochs, dev_loss, '-b', label='dev_loss')
plt.plot(epochs, train_loss, '-r', label='train_loss')
plt.xlabel('Epochs')


plt.legend()
plt.grid()

#save the plot
plt.savefig('loss.png')

