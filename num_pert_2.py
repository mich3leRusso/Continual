import os
import json
import argparse
import torch
import random
import torch.optim.lr_scheduler
from models import gresnet32, gresnet18, gresnet18mlp
from mind import MIND
from copy import deepcopy
from utils.generic import freeze_model, set_seed, setup_logger
from utils.publisher import push_results
from utils.transforms import to_tensor_and_normalize, default_transforms,default_transforms_core50,\
    to_tensor_and_normalize_core50,default_transforms_TinyImageNet,to_tensor_and_normalize_TinyImageNet, default_transforms_Synbols,to_tensor_and_normalize_Synbols, to_tensor
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from continuum import ClassIncremental
from continuum.tasks import split_train_val
from continuum.datasets import CIFAR100,Core50
from test_fn import test#, pert_CSI
import pickle as pkl
from parse import args
from utils.core50dset import get_all_core50_data, get_all_core50_scenario
from utils.tiny_imagenet_dset import get_all_tinyImageNet_data
from utils.synbols_dset import get_synbols_data
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario
import numpy as np
from time import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def get_stat_exp(y, y_hats, exp_idx, task_id, task_predictions):

    """ Compute accuracy and task accuracy for each experience."""
    conf_mat = torch.zeros((exp_idx+1, exp_idx+1))
    for i in range(exp_idx+1):
        ybuff= y[task_id==i]
        y_hats_buff=y_hats[task_id==i]
        acc = (ybuff==y_hats_buff).sum()/y_hats_buff.shape[0]

        for j in range(exp_idx+1):
            conf_mat[i,j] = ((task_id==i)&(task_predictions==j)).sum()/(task_id==i).sum()

        print(f"EXP:{i}, acc:{acc:.3f}, task:{conf_mat[i,i]:.3f}, distrib:{[round(conf_mat[i,j].item(), 3) for j in range(exp_idx+1)]}")


def entropy(vec):
    return -torch.sum(vec * torch.log(vec + 1e-7), dim=1)

# model
if args.model == 'gresnet32':
    model = gresnet32(dropout_rate = args.dropout)
elif args.model == 'gresnet18':
    model = gresnet18(num_classes=args.n_classes)
elif args.model == 'gresnet18mlp':
    model = gresnet18mlp(num_classes=args.n_classes)
else:
    raise ValueError("Model not found.")

data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/data_64x64')

# log files
setup_logger()

acc_ = []
num_exp = 21  #perchè il primo valore è la baseline
for ss in range(10):
    set_seed(ss)
    file_name = args.run_name[:-2]+f"_{ss}"
    print(file_name)
    model.load_state_dict(torch.load(f"/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/logs/{file_name}/checkpoints/weights.pt"))
    bn_weights = pkl.load(open(f"/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/logs/{file_name}/checkpoints/bn_weights.pkl", "rb"))
    model.bn_weights = bn_weights

    model.to(args.device)

    strategy = MIND(model)

    if args.dataset == 'CIFAR100':
        class_order = list(range(100))
        random.shuffle(class_order)

        train_dataset = CIFAR100(data_path, download=True, train=True)
        test_dataset = CIFAR100(data_path, download=True, train=False)

        r = int(args.extra_classes / args.classes_per_exp)

        if args.control == 1:
            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for k in range(r):
                for i in range(len(old_y)):
                    new_y.append(old_y[i])
                    new_y.append(old_y[i])
                    new_x.append(old_x[i])
                    new_x.append(np.rot90(old_x[i], k+1))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)

        if args.extra_classes > 0:
            # modifico il class order
            class_order_ = []
            for t in range(10):
                for k in range(r+1):
                    for c in range(10):
                        class_order_.append(class_order[t * 10 + c]+100*k)
            class_order = class_order_

            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for i in range(len(old_y)):
                for k in range(r+1):
                    new_y.append(old_y[i]+100*k)
                    new_x.append(np.rot90(old_x[i], k))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)

            # modifico i dati in se (test)
            new_y = []
            new_x = []
            old_x = test_dataset.get_data()[0]
            old_y = test_dataset.get_data()[1]

            #####################################################################
            #                 ATTENZIONE! Problema da risolvere                 #
            #####################################################################
            for i in range(args.n_experiences * args.extra_classes):
                new_y.append(100+i)
                new_x.append(old_x[0])
            #####################################################################

            for i in range(len(old_y)):
                new_y.append(old_y[i])
                #new_y.append(old_y[i] + 100)
                new_x.append(old_x[i])
                #new_x.append(np.rot90(old_x[i], 2))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            test_dataset = InMemoryDataset(new_x, new_y)

            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp + args.extra_classes,
                class_order=class_order,
                transformations=default_transforms)
        else:
            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp,
                class_order=class_order,
                transformations=default_transforms)

        inc = args.classes_per_exp + args.extra_classes

        tra = to_tensor

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=inc,
            class_order=class_order,
            transformations=tra)
        strategy.test_scenario_2 = ClassIncremental(
            test_dataset,
            increment=inc,
            class_order=class_order,
            transformations=to_tensor_and_normalize)

    elif args.dataset == 'TinyImageNet':
        data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/data_64x64')
        train_data, test_data = get_all_tinyImageNet_data(data_path, args.n_experiences)

        r = int(args.extra_classes / 20)

        if args.extra_classes > 0:
            new_x = []
            new_y = []
            new_z = []  # task di appartenenza

            old_x = train_data[0]
            old_y = train_data[1]
            old_z = train_data[2]

            for k in range(r + 1):
                for i in range(old_x.shape[0]):
                    new_x.append(np.rot90(old_x[i], k))
                    new_y.append(old_y[i] + 200 * k)
                    new_z.append(old_z[i])
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            new_z = np.array(new_z)

            class_order = []
            for i in range(args.n_experiences):
                classes_in_task = np.unique(new_y[new_z == i])
                for j in range(len(classes_in_task)):
                    class_order.append(int(classes_in_task[j]))

            train_data = (new_x, new_y)

            new_x = []
            new_y = []

            old_x = test_data[0]
            old_y = test_data[1]

            for i in range(old_x.shape[0]):
                new_x.append(old_x[i])
                new_y.append(old_y[i])
            for j in range(r):
                for i in range(200):
                    new_x.append(old_x[0])
                    new_y.append(200 * (j + 1) + i)

            new_x = np.array(new_x)
            new_y = np.array(new_y)

            test_data = (new_x, new_y)
        else:
            class_order = []
            for i in range(200):
                class_order.append(i)

        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        inc = args.n_classes // args.n_experiences + args.extra_classes

        tra = to_tensor

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            class_order=class_order,
            increment=inc,
            transformations=default_transforms_TinyImageNet)

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            class_order=class_order,
            increment=inc,
            transformations=tra)

        strategy.test_scenario_2 = ClassIncremental(
            test_dataset,
            class_order=class_order,
            increment=inc,
            transformations=to_tensor_and_normalize_TinyImageNet)

    elif 'CORE50' in args.dataset :
        data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/data_64x64/core50_128x128')
        if args.dataset == 'CORE50_CI':
            train_data, test_data = get_all_core50_data(data_path, args.n_experiences, split=0.8)
        else:
            train_data, test_data = get_all_core50_scenario(data_path, split=0.8)

        r = int(args.extra_classes / 5)

        if args.extra_classes > 0:
            new_x = []
            new_y = []
            new_z = []  # task di appartenenza

            old_x = train_data[0]
            old_y = train_data[1]
            old_z = train_data[2]

            for k in range(r + 1):
                for i in range(old_x.shape[0]):
                    new_x.append(np.rot90(old_x[i], k))
                    new_y.append(old_y[i] + 50 * k)
                    new_z.append(old_z[i])
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            new_z = np.array(new_z)

            class_order = []
            for i in range(args.n_experiences):
                classes_in_task = np.unique(new_y[new_z == i])
                for j in range(len(classes_in_task)):
                    class_order.append(int(classes_in_task[j]))

            train_data = (new_x, new_y)

            new_x = []
            new_y = []

            old_x = test_data[0]
            old_y = test_data[1]

            for i in range(old_x.shape[0]):
                new_x.append(old_x[i])
                new_y.append(old_y[i])
            for j in range(r):
                for i in range(50):
                    new_x.append(old_x[0])
                    new_y.append(50 * (j + 1) + i)

            new_x = np.array(new_x)
            new_y = np.array(new_y)

            test_data = (new_x, new_y)
        else:
            class_order = []
            for i in range(50):
                class_order.append(i)

        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        ### qui manca tutto il blocco per gestire le rotazioni

        inc = args.n_classes//args.n_experiences + args.extra_classes

        tra = to_tensor

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            class_order=class_order,
            increment=inc,
            transformations=default_transforms_core50)

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            class_order=class_order,
            increment=inc,
            transformations=tra)

        strategy.test_scenario_2 = ClassIncremental(
            test_dataset,
            class_order=class_order,
            increment=inc,
            transformations=to_tensor_and_normalize_core50)

    elif args.dataset == 'Synbols':

        train_data, test_data = get_synbols_data('/davinci-1/home/dmor/PycharmProjects/MIND/data_64x64', n_tasks=args.n_experiences)
        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        class_order = list(range(200))
        random.shuffle(class_order)

        r = int(args.extra_classes / args.classes_per_exp)

        if args.control == 1:
            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for k in range(r):
                for i in range(len(old_y)):
                    new_y.append(old_y[i])
                    new_y.append(old_y[i])
                    new_x.append(old_x[i])
                    new_x.append(np.rot90(old_x[i], k + 1))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)

        if args.extra_classes > 0:
            # modifico il class order
            class_order_ = []
            for t in range(10):
                for k in range(r + 1):
                    for c in range(20):
                        class_order_.append(class_order[t * 20 + c] + 200 * k)
            class_order = class_order_

            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for i in range(len(old_y)):
                for k in range(r + 1):
                    new_y.append(old_y[i] + 200 * k)
                    new_x.append(np.rot90(old_x[i], k))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)

            # modifico i dati in se (test)
            new_y = []
            new_x = []
            old_x = test_dataset.get_data()[0]
            old_y = test_dataset.get_data()[1]

            #####################################################################
            #                 ATTENZIONE! Problema da risolvere                 #
            #####################################################################
            for i in range(args.n_experiences * args.extra_classes):
                new_y.append(200 + i)
                new_x.append(old_x[0])
            #####################################################################

            for i in range(len(old_y)):
                new_y.append(old_y[i])
                # new_y.append(old_y[i] + 100)
                new_x.append(old_x[i])
                # new_x.append(np.rot90(old_x[i], 2))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            test_dataset = InMemoryDataset(new_x, new_y)

            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp + args.extra_classes,
                class_order=class_order,
                transformations=default_transforms_Synbols)
        else:
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            permutazione = sorted(range(len(old_y)), key=lambda i: old_y[i])
            new_x = old_x[permutazione]
            new_y = old_y[permutazione]
            train_dataset = InMemoryDataset(new_x, new_y)

            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp,
                class_order=class_order,
                transformations=default_transforms_Synbols)

        inc = args.classes_per_exp + args.extra_classes

        tra = to_tensor

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=inc,
            class_order=class_order,
            transformations=tra)
        strategy.test_scenario_2 = ClassIncremental(
            test_dataset,
            increment=inc,
            class_order=class_order,
            transformations=to_tensor_and_normalize)

    print(f"Number of classes: {strategy.train_scenario.nb_classes}.")
    print(f"Number of tasks: {strategy.train_scenario.nb_tasks}.")

    strategy.pruner.masks = torch.load(f"/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/logs/{file_name}/checkpoints/masks.pt")

    for i, train_taskset in enumerate(strategy.train_scenario):
        if args.packnet_original:
            with torch.no_grad():
                strategy.pruner.dezero(strategy.model)

        strategy.experience_idx = i
        strategy.model.set_output_mask(i, train_taskset.get_classes())

        model.load_bn_params(strategy.experience_idx)

        # prepare dataset
        strategy.train_taskset, strategy.val_taskset = split_train_val(train_taskset, val_split=args.val_split)
        strategy.train_dataloader = DataLoader(strategy.train_taskset, batch_size=args.bsize, shuffle=True)
        if len(strategy.val_taskset):
            strategy.val_dataloader = DataLoader(strategy.val_taskset, batch_size=args.bsize, shuffle=True)
        else:
            strategy.val_dataloader = DataLoader(strategy.test_scenario_2[i], batch_size=args.bsize, shuffle=True)

        #################### TEST ##########################
        if i!= 9:
            continue
        # concatenate pytorch datasets up to the current experience
        default_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor()
        ]
        with torch.no_grad():
            acc = []
            acc_e = []
            taw = []
            cl = []

            test_set=strategy.test_scenario[:i+1]
            strategy.model.eval()
            dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

            hist_1 = []
            hist_2 = []
            if args.extra_classes == 0:
                num_rot = 1
            else:
                num_rot = int(args.extra_classes/args.classes_per_exp)+1

            if args.aug_inf == 0:
                num_rot = 1

            for k in range(num_exp): #run sulle varie aumentazioni
                print(f"k = {k}")
                rot = k % num_rot

                if args.extra_classes > 0:
                    s = args.n_classes + int(args.extra_classes * args.n_experiences)
                    confusion_mat = torch.zeros((s, s))
                    confusion_mat_e = torch.zeros((s, s))
                    confusion_mat_taw = torch.zeros((s, s))
                else:
                    confusion_mat = torch.zeros((args.n_classes, args.n_classes))
                    confusion_mat_e = torch.zeros((args.n_classes, args.n_classes))
                    confusion_mat_taw = torch.zeros((args.n_classes, args.n_classes))

                y_hats = []
                y_hats_e = []
                y_taw = []
                ys = []
                task_predictions = []
                task_ids = []
                for i, (x, y, task_id) in enumerate(dataloader):
                    frag_preds = []
                    entropy_frag = []
                    for j in range(strategy.experience_idx + 1):
                        # create a temporary model copy
                        model = freeze_model(deepcopy(strategy.model))

                        strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
                        model.load_bn_params(j)
                        model.exp_idx = j

                        if k != 0:
                            if args.dataset == 'TinyImageNet':
                                trans = transforms.Compose([
                                    transforms.RandomCrop(64, padding=8),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=63 / 255),
                                    transforms.Normalize(
                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                    ),
                                ])
                            elif args.dataset == 'Synbols':
                                trans = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(
                                        (0.47957573372395784, 0.4786930207950382, 0.4795725401730997), (0.2840923785401597, 0.28447272496390646, 0.28412646131981306)
                                    )
                                ])
                            elif args.dataset == 'CORE50_CI':
                                trans = transforms.Compose([
                                    transforms.RandomCrop(64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=.5,hue=.3),

                                    transforms.Normalize((0.5998523831367493, 0.5575963854789734, 0.5395311713218689), (0.20457075536251068, 0.2166813313961029, 0.22945666313171387)
                                    ),
                                ])
                            elif args.dataset == 'CIFAR100':
                                trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                                        transforms.ColorJitter(brightness=63 / 255),
                                                        transforms.Normalize((0.5071, 0.4865, 0.4409),
                                                                             (0.2673, 0.2564, 0.2762))])

                            pred = []
                            x_ = []
                            for img in x:
                                img = np.array(img)
                                if (args.dataset=="CORE50_CI") | (args.dataset=="Synbols"):
                                    img = np.rot90(img.transpose(2, 1, 0), -rot).transpose(2, 1, 0)
                                else:
                                    img = np.rot90(img.transpose(2, 1, 0), rot).transpose(2, 1, 0)
                                img = torch.tensor(img.copy())
                                new_img = trans(img)
                                x_.append(new_img)
                            x_2 = torch.stack(x_)
                            pred=model(x_2.to(args.device))
                        else:
                            if args.dataset == 'TinyImageNet':
                                trans = transforms.Compose(
                                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                            if args.dataset == 'Synbols':
                                trans = transforms.Compose([
                                    transforms.Normalize((0.47957573372395784, 0.4786930207950382, 0.4795725401730997), (0.2840923785401597, 0.28447272496390646, 0.28412646131981306))])
                            elif args.dataset == 'CORE50_CI':
                                trans = transforms.Compose(
                                    [transforms.Normalize((0.5998523831367493, 0.5575963854789734, 0.5395311713218689), (0.20457075536251068, 0.2166813313961029, 0.22945666313171387))])
                            else:
                                trans = transforms.Compose(
                                    [transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
                            x_ = []
                            for img in x:
                                x_.append(trans(img))
                            x_2 = torch.stack(x_)
                            pred = model(x_2.to(args.device))
                        #else:
                        #    pred = model(x.to(args.device))


                        if args.extra_classes > 0:
                            pred = pred[:, j * (args.classes_per_exp + args.extra_classes): (j + 1) * (
                                        args.classes_per_exp + args.extra_classes)]
                        else:
                            pred = pred[:, j * args.classes_per_exp:(j + 1) * args.classes_per_exp]

                        # nella modalità 4 abbiamo classi aggiuntive quindi rimuovo i relativi pezzi ####################
                        if args.extra_classes > 0:
                            sp = torch.softmax(pred / args.temperature, dim=1)
                            sp = sp[:, args.classes_per_exp*rot:args.classes_per_exp*(rot+1)]
                            frag_preds.append(torch.softmax(sp / args.temperature, dim=1))
                        else:
                            frag_preds.append(torch.softmax(pred / args.temperature, dim=1))
                        entropy_frag.append(entropy(torch.softmax(pred / args.temperature, dim=1)))

                    # on_shell_confidence, elsewhere_confidence = confidence(frag_preds, task_id)
                    # print(f"on_shell confidence:{[round(c.item(), 2) for c in on_shell_confidence]}\nelsewhere confidence:{[round(c.item(), 2) for c in elsewhere_confidence]}")

                    frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]
                    entropy_frag = torch.stack(entropy_frag)  # [bsize, n_frag]

                    if k == 1:
                        hist_1.append(frag_preds)
                        hist_2.append(entropy_frag)
                    elif k > 1:
                        frag_preds = (frag_preds + hist_1[i]*(k-1))/k
                        entropy_frag = (entropy_frag + hist_2[i] * (k-1)) / k

                        hist_1[i] = frag_preds
                        hist_2[i] = entropy_frag


                    batch_size = frag_preds.shape[1]

                    ### select across the top 2 of likelihood the head  with the lowest entropy
                    # buff -> batch_size  x 2, 0-99 val
                    # print(frag_preds[:, 9, :])
                    buff = frag_preds.max(dim=-1)[0].argsort(dim=0)[-2:]  # [2, bsize]

                    # buff_entropy ->  2 x batch_size, entropy values
                    indices = torch.arange(batch_size)
                    if buff.shape[0] == 1:
                        buff_entropy = entropy_frag[buff[0, :], indices].unsqueeze(0)
                    else:
                        a = entropy_frag[buff[0, :], indices]
                        b = entropy_frag[buff[1, :], indices]
                        buff_entropy = torch.stack((a, b))  # [2, bsize]

                    # index of min entropy ->  1 x batch_size, val 0-1
                    index_min = buff_entropy.argsort(dim=0)[0, :]

                    index_class = buff[index_min, indices]

                    task_predictions.append(buff[-1])
                    if args.dataset == 'CORE50':
                        y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1))
                        y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1))
                        y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1))

                    else:
                        if args.extra_classes == 0:
                            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1) + args.classes_per_exp * index_class)
                            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1) + args.classes_per_exp * buff[-1])
                            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + (
                                        args.classes_per_exp * task_id.to(args.cuda)).to(torch.int32))
                        else:
                            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1) + (
                                        args.classes_per_exp + args.extra_classes) * index_class)
                            y_hats.append(
                                frag_preds[buff[-1], indices].argmax(dim=1) + (args.classes_per_exp + args.extra_classes) *
                                buff[-1])
                            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + (
                                        (args.classes_per_exp + args.extra_classes) * task_id.to(args.cuda)).to(torch.int32))

                    task_ids.append(task_id)
                    ys.append(y)

                # concat labels and preds
                y_hats = torch.cat(y_hats, dim=0).to('cpu')
                y_hats_e = torch.cat(y_hats_e, dim=0).to('cpu')
                y = torch.cat(ys, dim=0).to('cpu')
                y_taw = torch.cat(y_taw, dim=0).to('cpu')
                task_predictions = torch.cat(task_predictions, dim=0).to('cpu')
                task_ids = torch.cat(task_ids, dim=0).to('cpu')

                a = y%(args.classes_per_exp + args.extra_classes)
                y = y[a < args.classes_per_exp]
                y_hats = y_hats[a < args.classes_per_exp]
                y_taw = y_taw[a < args.classes_per_exp]
                y_hats_e = y_hats_e[a < args.classes_per_exp]
                task_predictions = task_predictions[a < args.classes_per_exp]
                task_ids = task_ids[a < args.classes_per_exp]





                # assign +1 to the confusion matrix for each prediction that matches the label
                for i in range(y.shape[0]):
                    confusion_mat[y[i], y_hats[i]] += 1
                    confusion_mat_e[y[i], y_hats_e[i]] += 1
                    confusion_mat_taw[y[i], y_taw[i]] += 1

                # task confusion matrix and forgetting mat
                if args.extra_classes == 0:
                    for j in range(strategy.experience_idx + 1):
                        i = strategy.experience_idx
                        acc_conf_mat_task = confusion_mat[j * args.classes_per_exp:(j + 1) * args.classes_per_exp,
                                            j * args.classes_per_exp:(
                                                                                 j + 1) * args.classes_per_exp].diag().sum() / confusion_mat[
                                                                                                                               i * args.classes_per_exp:(
                                                                                                                                                                    i + 1) * args.classes_per_exp,
                                                                                                                               :].sum()
                        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
                        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max() - acc_conf_mat_task
                else:
                    # task confusion matrix and forgetting mat
                    for j in range(strategy.experience_idx + 1):
                        i = strategy.experience_idx
                        acc_conf_mat_task = confusion_mat[j * (args.classes_per_exp + args.extra_classes):(j + 1) * (
                                    args.classes_per_exp + args.extra_classes),
                                            j * (args.classes_per_exp + args.extra_classes):(j + 1) * (
                                                        args.classes_per_exp + args.extra_classes)].diag().sum() / confusion_mat[
                                                                                                                   i * (
                                                                                                                               args.classes_per_exp + args.extra_classes):(
                                                                                                                                                                                      i + 1) * (
                                                                                                                                                                                      args.classes_per_exp + args.extra_classes),
                                                                                                                   :].sum()
                        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
                        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max() - acc_conf_mat_task

                # compute accuracy
                #plt.imshow(confusion_mat, cmap='viridis', interpolation='nearest')
                #plt.show()

                accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
                accuracy_e = confusion_mat_e.diag().sum() / confusion_mat_e.sum()
                accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()

                task_accuracy = (task_predictions == task_ids).sum() / y_hats.shape[0]

                acc.append(accuracy.item())
                acc_e.append(accuracy_e.item())
                taw.append(accuracy_taw.item())
                cl.append(task_accuracy.item())
                #print(accuracy.item())

    print(acc)
    print(taw)
    acc_.append(acc)