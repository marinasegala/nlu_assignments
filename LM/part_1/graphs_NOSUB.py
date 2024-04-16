import csv
import matplotlib.pyplot as plt
import numpy as np

def create_array(index):
    array = []
    count = 0
    inputfile = csv.reader(open('results_1.csv','r'))
    for row in inputfile:  
        if row[index] != '' and count != 0:
            array.append(int(float(row[index])))
        count += 1
    return array



epochs = create_array(0)
train_loss = create_array(1)
dev_loss = create_array(2)
pll = create_array(3)
markers_res_0 = [12, 29, 31, 33, 35, 37, 40, 41, 43, 44, 45]
markers_res_1 = [30, 48, 53, 61, 63, 68, 72, 75, 76, 77]
print(len(epochs))
print(len(pll))

#plt.plot(epochs, pll, '-b.', markevery=markers, label='pll')
#plt.plot(epochs, dev_loss, '-r', label='train_loss')
plt.xlabel('Epochs')


plt.legend()
plt.grid()

plt.show()

