import re
import numpy as np
import os
import matplotlib.pyplot as plt
import ast


TAG_1 = []
TAW_1 = []

TAG_2 = []
TAW_2 = []

path = '/davinci-1/home/dmor/'

files = [f for f in os.listdir(path) if f.startswith('cifar100_CA2_rot.o')]
nome_file = path + files[0]
p=0
tag = []
taw = []
with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        if riga[0] == 'n':
            numeri = re.findall(r'-?\d+(?:[\.,]\d+)?', riga)
            tag.append(float(numeri[1]))
            taw.append(float(numeri[3]))
        if len(tag) == 31:
            TAG_1.append(tag)
            TAW_1.append(taw)
            tag = []
            taw = []
TAG_1 = np.array(TAG_1)
TAW_1 = np.array(TAW_1)

files = [f for f in os.listdir(path) if f.startswith('cifar100_CA2_rot.o')]
nome_file = path + files[0]
p=0
tag = []
taw = []
with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        if riga[0] == 'n':
            numeri = re.findall(r'-?\d+(?:[\.,]\d+)?', riga)
            tag.append(float(numeri[1]))
            taw.append(float(numeri[3]))
        if len(tag) == 31:
            TAG_2.append(tag)
            TAW_2.append(taw)
            tag = []
            taw = []
TAG_2 = np.array(TAG_2)
TAW_2 = np.array(TAW_2)
tag = []
taw = []

TAG_1 = TAG_1[9, :]
TAG_2 = TAG_2[m, :]

plt.plot(range(1,len(TAG_1)), TAG_1[1:], color='blue')
plt.plot(range(1,len(TAG_2)), TAG_2[1:], color='green')
plt.plot(range(1,len(TAG_1)), np.ones(len(TAG_1)-1)*TAG_1[0], color='red')
plt.show()

