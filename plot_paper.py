import matplotlib.pyplot as plt


#Seleziona grafico
j = 3

# Titolo del grafico
titles = ["CIFAR100", "TinyImageNet", "Core50", "Synbols", "CIFAR100", "TinyImageNet", "Core50", "Synbols", "Control Data Augmentation", "Different types of augmentation"]

# Dati con valore ± varianza
plot = [
    #TAG
    [["39.7 ±  0.7",   "43.8 ±  1.2",  "44.7 ±  0.6",  "44.5 ±  0.9"],
    ["42.7 ±  0.8",   "47.0 ±  1.2",  "48.4 ±  0.5",	"48.2 ±  0.8"],
    ["42.7 ±  0.8",   "47.5 ±  1.0",	"51.0 ±  0.7",	"51.0 ±  0.9"]],

    [["34.7 ±  0.9",	"37.0 ±  0.8",	"38.1 ±  1.0",	"37.2 ±  0.7"],
    ["37.6 ±  0.9",	"40.2 ±  0.8",	"41.3 ±  0.9",	"40.5 ±  0.7"],
    ["37.6 ±  0.9",  "41.2 ±  1.1",  "43.5 ±  0.6",   "42.9 ±  0.9"]],

    [["57.5 ±  1.8",	"61.3 ±  2.1",	"58.6 ±  2.5",	"56.6 ±  1.0"],
    ["64.5 ±  1.6",	"69.8 ±  1.8",	"67.5 ±  2.8",	"66.3 ±  0.8"],
    ["64.5 ±  1.6",  "76.5 ±  1.6",	"77.2 ±  2.9",	"79.7 ±  0.7"]],

    [["77.63 ± 2.00",	"82.93 ± 1.74",	"85.53 ± 2.42",	"86.59 ± 0.99"],
    ["82.79 ± 1.78",	    "87.36 ± 1.60",	"88.97 ± 2.15",	"90.05 ± 0.62"],
    ["82.79 ± 1.78",      "92.31 ± 1.08",	"95.22 ± 0.80",	"96.04 ± 0.46"]],

    #TAw
    [["82.19 ±  0.48",	"83.68 ±  0.67",	"84.03 ±  0.67",	"84.61 ±  0.68"],
    ["83.82 ±  0.67",	"85.26 ±  0.80",	"85.77 ±  0.45",	"86.23 ±  0.58"],
    ["83.82 ±  0.67",  "85.35 ±  0.79",	    "86.86 ±  0.52",	"87.27 ±  0.46"]],

    [["71.10 ±  0.45",	"72.71 ±  0.51",  "73.69 ±  0.51",	"73.35 ±  0.41"],
    ["73.38 ±  0.51",	"75.12 ±  0.46",	"75.90 ±  0.49",	"75.49 ±  0.40"],
    ["73.38 ±  0.51",  "75.77 ±  0.47",	"77.14 ±  0.44",	"77.01 ±  0.41"]],

    [["99.71 ±  0.15",	"99.79 ±  0.08",	"99.72 ±  0.11",	"99.66 ±  0.13"],
    ["99.83 ±  0.11",	"99.90 ±  0.05",	"99.87 ±  0.06",	"99.83 ±  0.08"],
    ["99.83 ±  0.11",   "99.94 ±  0.03",	"99.92 ±  0.05",	"99.92 ±  0.05"]],

    [["98.35 ±  0.27",	 "98.91 ±  0.15",	"99.18 ± 0.13",	"99.16 ± 0.10"],
    ["98.89 ± 0.19",	 "99.24 ± 0.14",	"99.41 ± 0.09",	"99.40 ± 0.04"],
    ["98.89 ± 0.19",     "99.54 ± 0.07",	"99.67 ± 0.04",	"99.71 ± 0.05"]],

    #Controlli
    [["39.7 ±  0.7",   "43.8 ±  1.2",  "44.7 ±  0.6",  "44.5 ±  0.9"],
    ["39.7 ±  0.7",	"37.3 ±  1.0",	"35.6 ±  0.4",	"34.3 ±  0.8"],
    ["39.7 ±  0.7",	"42.5 ±  1.3",	"42.5 ±  0.7",	"42.2 ±  1.4"]],

    #TTDA
    [["39.7 ±  0.7",	"43.8 ±  1.2",	    "44.7 ±  0.6",	"44.5 ±  0.9"],
    ["41.5 ± 0.7",	    "46.6 ± 1.0",	    "50.6 ± 0.7",	"50.2 ± 0.9"],
    ["42.7 ±  0.8",	    "47.5 ±  1.0",	    "51.0 ±  0.7",	"51.0 ±  0.9"],
    ["41.5 ± 0.7",	    "45.3 ± 1.3",	    "46.8 ± 0.7",	"46.4 ± 0.9"],
    ["39.7 ±  0.7",	    "45.5 ± 1.0",	    "49.7 ± 0.7",	"49.4 ± 0.9"],
    ["39.8 ± 0.7",	    "43.9 ± 1.3",	    "44.8 ± 0.5",	"44.7 ± 1.0"],
    ["41.5 ± 0.7",	    "45.8 ± 1.0",	    "46.8 ± 0.7",	"46.9 ± 0.9"]]
]

for j in range(len(plot)):
    # Etichette delle curve
    if j < 8:
        row_labels = ["MIND", "TTDA", "TTDA + rotations"]
    elif j == 8:
        row_labels = ["Class augmentation", "Data augmentation", "Data augmentation with artificial classes"]
    elif j  ==  9:
        row_labels = ["No augmentations", "rotations + flip", "All augmentations", "flip", "rotations", "color Jittering", "random crop"]

    # Parsing dei valori principali e delle varianze
    means = []
    stds = []

    for row in plot[j]:
        mean_row = []
        std_row = []
        for cell in row:
            mean_str, std_str = cell.split("±")
            mean_row.append(float(mean_str.strip()))
            std_row.append(float(std_str.strip()))
        means.append(mean_row)
        stds.append(std_row)

    # X: 1-based index
    x_vals = [1, 2, 3, 4]

    # Plotting con ombre di varianza
    plt.figure(figsize=(8, 5))
    for mean, std, label in zip(means, stds, row_labels):
        #plt.plot(x_vals, mean, marker='o', label=label)
        #lower = [m - s for m, s in zip(mean, std)]
        #upper = [m + s for m, s in zip(mean, std)]
        #plt.fill_between(x_vals, lower, upper, alpha=0.2)
        plt.errorbar(x_vals, mean, yerr=std, fmt='-o', capsize=5, label=label)

    # Asse x: solo interi
    plt.xticks(ticks=x_vals, labels=["x1", "x2", "x3", "x4"])
    if j == 7:
        y_vals = [98.0, 98.5, 99.0, 99.5, 100]
        plt.yticks(ticks=y_vals)
    plt.xlabel('Class augmentation')
    if (j < 4) | (j>=8):
        plt.ylabel('Accuracy TAG (%)')
    else:
        plt.ylabel('Accuracy TAW (%)')
    plt.title(titles[j])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()