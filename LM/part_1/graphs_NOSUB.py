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
dim = [40,47,51,53,55,64,67,73,75,82,83,86,88,89,91,94,98]

print(len(epochs))
print(len(pll))

plt.plot(epochs, pll, label='Pll')
for d in dim:
    plt.plot(epochs[d], pll[d], 'ro')
plt.xlabel('Epochs')


plt.legend()
plt.grid()

plt.show()

