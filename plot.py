import re
import numpy as np
import os
import matplotlib.pyplot as plt
import ast


acc = []
acc_TAG = []

path = '/davinci-1/home/dmor/'
files = [f for f in os.listdir(path) if f.startswith('plot_tiny_CSIx2_ruotato.o')]
print(files)
nome_file = path + files[0]
p=0
with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        print(riga)
        s = riga.strip()
        if s[0] == '[':
            lista = ast.literal_eval(s)
            if p % 2 == 1:
                acc.append(lista)
            else:
                acc_TAG.append(lista)
            p += 1

acc = np.array(acc)
medie = np.mean(acc.T, axis=1)
std = np.std(acc.T, axis=1)

acc_TAG = np.array(acc_TAG)
medie_TAG = np.mean(acc_TAG.T, axis=1)
std_TAG = np.std(acc_TAG.T, axis=1)

print(acc.shape)
print('\n')

'''plt.plot(range(1,len(medie)), medie[1:], color='blue')
plt.plot(range(1,len(medie)), medie[1:]+std[1:], color='deepskyblue')
plt.plot(range(1,len(medie)), medie[1:]-std[1:], color='deepskyblue')

plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*medie[0], color='red')
plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*(medie[0]+std[0]), color='salmon')
plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*(medie[0]-std[0]), color='salmon')'''

print(f"{medie[0]*100:.1f} ± {std[0]*100:.1f}")
print(f"{medie_TAG[0]*100:.1f} ± {std_TAG[0]*100:.1f}")
print('\n')

print(f"{medie[10]*100:.1f} ± {std[10]*100:.1f}")
print(f"{medie_TAG[10]*100:.1f} ± {std_TAG[10]*100:.1f}")
print('\n')

print(f"{medie[20]*100:.1f} ± {std[20]*100:.1f}")
print(f"{medie_TAG[20]*100:.1f} ± {std_TAG[20]*100:.1f}")


#plt.show()