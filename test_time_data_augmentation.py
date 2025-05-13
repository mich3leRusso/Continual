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
from utils.transforms import ttda_cifar100, normalize_cifar100, ttda_core50, normalize_core50, ttda_TinyImageNet, normalize_TinyImageNet, ttda_Synbols, normalize_Synbols, to_tensor
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

#data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/data_64x64')
project_path = '/davinci-1/home/dmor/PycharmProjects/MIND'
data_path = project_path + '/data'

model = gresnet32(dropout_rate = args.dropout)

# log files
setup_logger()

acc_ = []
taw_ = []
num_exp = args.n_aug + 1  #perchè il primo valore è la baseline
for seed in range(10):
    set_seed(seed)
    file_name = args.run_name[:-2]+f"_{seed}"
    print(file_name)
    model.load_state_dict(torch.load(project_path + f"/logs/{file_name}/checkpoints/weights.pt"))
    bn_weights = pkl.load(open(project_path + f"/logs/{file_name}/checkpoints/bn_weights.pkl", "rb"))
    model.bn_weights = bn_weights
    model.to(args.device)
    strategy = MIND(model)

    if args.dataset == 'CIFAR100':
        train_dataset = CIFAR100(data_path, download=True, train=True).get_data()
        test_dataset = CIFAR100(data_path, download=True, train=False).get_data()
        transform_0 = ttda_cifar100
        transform_1 = normalize_cifar100
    elif args.dataset == 'CORE50_CI':
        data_path = os.path.expanduser(data_path + '/core50_128x128')
        train_dataset, test_dataset = get_all_core50_data(data_path, args.n_experiences, split=0.8)
        transform_0 = ttda_core50
        transform_1 = normalize_core50
    elif args.dataset == 'TinyImageNet':
        data_path = os.path.expanduser(data_path)
        train_dataset, test_dataset = get_all_tinyImageNet_data(data_path, args.n_experiences)
        transform_0 = ttda_TinyImageNet
        transform_1 = normalize_TinyImageNet
    elif args.dataset == 'Synbols':
        train_data, test_data = get_synbols_data(data_path, n_tasks=args.n_experiences)
        train_dataset = InMemoryDataset(*train_data).get_data()
        test_dataset = InMemoryDataset(*test_data).get_data()
        transform_0 = ttda_Synbols
        transform_1 = normalize_Synbols
    transform_2 = to_tensor

    r = args.class_augmentation - 1

    if (args.dataset == 'CIFAR100') | (args.dataset == 'TinyImageNet'):
        class_order = list(range(args.n_classes))
        random.shuffle(class_order)
        class_order_ = []
        for t in range(args.n_experiences):
            for k in range(r + 1):
                for c in range(args.classes_per_exp):
                    class_order_.append(class_order[t * args.classes_per_exp + c] + args.n_classes * k)
        class_order = class_order_

    # modifying test set
    new_y = []
    new_x = []
    old_x = test_dataset[0]
    old_y = test_dataset[1]
    for i in range(args.n_experiences * args.extra_classes):
        new_y.append(args.n_classes + i)
        new_x.append(old_x[0])
    for i in range(len(old_y)):
        new_y.append(old_y[i])
        new_x.append(old_x[i])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    test_dataset = InMemoryDataset(new_x, new_y)

    if (args.dataset == 'CORE50_CI') | (args.dataset == 'Synbols'):
        class_order = []
        for i in range(args.n_experiences):
            classes_in_task = np.unique(new_y[new_z == i])
            for j in range(len(classes_in_task)):
                class_order.append(int(classes_in_task[j]))

    strategy.test_scenario = ClassIncremental(
        test_dataset,
        increment=args.classes_per_exp + args.extra_classes,
        class_order=class_order,
        transformations=transform_2)

    strategy.pruner.masks = torch.load(project_path + f"/logs/{file_name}/checkpoints/masks.pt")

    for i, test_taskset in enumerate(strategy.test_scenario):
        if args.packnet_original:
            with torch.no_grad():
                strategy.pruner.dezero(strategy.model)

        strategy.experience_idx = i
        strategy.model.set_output_mask(i, test_taskset.get_classes())

        model.load_bn_params(strategy.experience_idx)

    with torch.no_grad():
        acc = []
        taw = []

        test_set=strategy.test_scenario[:9+1]
        strategy.model.eval()
        dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

        hist_1 = []
        if args.extra_classes == 0:
            num_rot = 1
        else:
            num_rot = int(args.extra_classes/args.classes_per_exp)+1

        if args.with_rotations == 0:
            num_rot = 1

        for k in range(num_exp): #run sulle varie aumentazioni
            print(f"augmentation_number = {k}")
            rot = k % num_rot

            if args.extra_classes > 0:
                s = args.n_classes + int(args.extra_classes * args.n_experiences)
                confusion_mat = torch.zeros((s, s))
                confusion_mat_taw = torch.zeros((s, s))
            else:
                confusion_mat = torch.zeros((args.n_classes, args.n_classes))
                confusion_mat_taw = torch.zeros((args.n_classes, args.n_classes))

            y_hats = []
            y_taw = []
            ys = []
            task_ids = []
            for i, (x, y, task_id) in enumerate(dataloader):
                frag_preds = []
                for j in range(strategy.experience_idx + 1):
                    # create a temporary model copy
                    model = freeze_model(deepcopy(strategy.model))

                    strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
                    model.load_bn_params(j)
                    model.exp_idx = j

                    if k != 0:
                        trans = transforms.Compose(transform_0)
                    else:
                        trans = transforms.Compose(transform_1)

                    x_ = []
                    for img in x:
                        img = np.array(img)
                        if (args.dataset == "CORE50_CI") | (args.dataset == "Synbols"):
                            img = np.rot90(img.transpose(2, 1, 0), -rot).transpose(2, 1, 0)
                        else:
                            img = np.rot90(img.transpose(2, 1, 0), rot).transpose(2, 1, 0)
                        img = torch.tensor(img.copy())
                        new_img = trans(img)
                        x_.append(new_img)
                    x_2 = torch.stack(x_)
                    pred = model(x_2.to(args.device))


                    pred = pred[:, j * (args.classes_per_exp + args.extra_classes): (j + 1) * (args.classes_per_exp + args.extra_classes)]

                    # removing scores associated with extra classes
                    sp = torch.softmax(pred / args.temperature, dim=1)
                    sp = sp[:, args.classes_per_exp*rot:args.classes_per_exp*(rot+1)]
                    frag_preds.append(torch.softmax(sp / args.temperature, dim=1))

                frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]

                if k == 1:
                    hist_1.append(frag_preds)
                elif k > 1:
                    frag_preds = (frag_preds + hist_1[i]*(k-1))/k
                    hist_1[i] = frag_preds

                batch_size = frag_preds.shape[1]

                ### select across the top 2 of likelihood the head  with the lowest entropy
                # buff -> batch_size  x 2, 0-99 val
                buff = frag_preds.max(dim=-1)[0].argsort(dim=0)[-2:]  # [2, bsize]

                # buff_entropy ->  2 x batch_size, entropy values
                indices = torch.arange(batch_size)

                y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1) + (args.classes_per_exp + args.extra_classes) *buff[-1])
                y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + ((args.classes_per_exp + args.extra_classes) * task_id.to(args.cuda)).to(torch.int32))

                task_ids.append(task_id)
                ys.append(y)

            # concat labels and preds
            y_hats = torch.cat(y_hats, dim=0).to('cpu')
            y = torch.cat(ys, dim=0).to('cpu')
            y_taw = torch.cat(y_taw, dim=0).to('cpu')
            task_ids = torch.cat(task_ids, dim=0).to('cpu')

            #to filter out the fake elements added before
            a = y%(args.classes_per_exp + args.extra_classes)
            y = y[a < args.classes_per_exp]
            y_hats = y_hats[a < args.classes_per_exp]
            y_taw = y_taw[a < args.classes_per_exp]
            task_ids = task_ids[a < args.classes_per_exp]

            # assign +1 to the confusion matrix for each prediction that matches the label
            for i in range(y.shape[0]):
                confusion_mat[y[i], y_hats[i]] += 1
                confusion_mat_taw[y[i], y_taw[i]] += 1

            # task confusion matrix and forgetting mat
            for j in range(strategy.experience_idx + 1):
                i = strategy.experience_idx
                acc_conf_mat_task = confusion_mat[j * (args.classes_per_exp + args.extra_classes):(j + 1) * (args.classes_per_exp + args.extra_classes),j * (args.classes_per_exp + args.extra_classes):(j + 1) * (args.classes_per_exp + args.extra_classes)].diag().sum() / confusion_mat[i * (args.classes_per_exp + args.extra_classes):(i + 1) * (args.classes_per_exp + args.extra_classes),:].sum()
                strategy.confusion_mat_task[i][j] = acc_conf_mat_task

            accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
            accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()

            acc.append(accuracy.item())
            taw.append(accuracy_taw.item())
    acc_.append(acc)
    taw_.append(taw)

    print(f"SEED: {seed}")
    tag_mean = np.mean(np.array(acc_).T, axis=1)
    tag_std = np.std(np.array(acc_).T, axis=1)
    taw_mean = np.mean(np.array(taw_).T, axis=1)
    taw_std = np.std(np.array(taw_).T, axis=1)

    point_of_interest = [0, int(args.n_aug/2), args.n_aug]
    for p in point_of_interest:
        print(f"number of augmentation = {p},     TAG = {tag_mean[p]*100:.2f} ± {tag_std[p]*100:.2f},    TAW = {taw_mean[p]*100:.2f} ± {taw_std[p]*100:.2f}")
