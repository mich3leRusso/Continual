import re
import numpy as np
import os
import matplotlib.pyplot as plt
import ast


medie = []
medie_TAG = []
std = []
std_TAG = []

path = '/davinci-1/home/dmor/'
files = [f for f in os.listdir(path) if f.startswith('cifar_100_control_2_CA2_rot.o')]
print(files)
nome_file = path + files[0]
p=0
with open(nome_file, 'r', encoding='utf-8') as file:
    for riga in file:
        if riga[0] == 'n':
            numeri = re.findall(r'-?\d+(?:[\.,]\d+)?', riga)
            medie.append(float(numeri[1]))
            std.append(float(numeri[2]))
            medie_TAG.append(float(numeri[3]))
            std_TAG.append(float(numeri[4]))
        if riga[0] == 'S':
            medie = []
            medie_TAG = []
            std = []
            std_TAG = []

'''plt.plot(range(1,len(medie)), medie[1:], color='blue')
plt.plot(range(1,len(medie)), medie[1:]+std[1:], color='deepskyblue')
plt.plot(range(1,len(medie)), medie[1:]-std[1:], color='deepskyblue')

plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*medie[0], color='red')
plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*(medie[0]+std[0]), color='salmon')
plt.plot(range(1,len(medie)), np.ones(len(medie)-1)*(medie[0]-std[0]), color='salmon')'''

l = len(medie)-1

print("no augmentation")
print(f"{medie[0]:.1f} ± {std[0]:.1f}")
print(f"{medie_TAG[0]:.1f} ± {std_TAG[0]:.1f}")
print('\n')

print(f"{int(l/2)} augmentations")
print(f"{medie[int(l/2)]:.1f} ± {std[int(l/2)]:.1f}")
print(f"{medie_TAG[int(l/2)]:.1f} ± {std_TAG[int(l/2)]:.1f}")
print('\n')

print(f"{l} augmentations")
print(f"{medie[l]:.1f} ± {std[l]:.1f}")
print(f"{medie_TAG[l]:.1f} ± {std_TAG[l]:.1f}")


#plt.show()