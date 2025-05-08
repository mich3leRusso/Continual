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
import torchvision.transforms.functional as F

if args.mode == 3:
    class RandomRotate180:
        def __call__(self, img):
            if random.random() < 0.5:
                return F.rotate(img, 180)
            return img

    default_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            RandomRotate180(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ]

def main():
    data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/MIND/data_64x64')

    # set seed
    set_seed(args.seed)

    # print
    if args.load_model_from_run != '':
        print("#"*30 + f"\n{args.run_name +' loads: '+args.load_model_from_run:^30}\n" +"#"*30)
    else:
        print("#"*30 + f"\n{args.run_name:^30}\n" +"#"*30)

    # log files 
    setup_logger()

    # model
    if args.model == 'gresnet32':
        model = gresnet32(dropout_rate = args.dropout)
    elif args.model == 'gresnet18':
        model = gresnet18(num_classes=args.n_classes)
    elif args.model == 'gresnet18mlp':
        model = gresnet18mlp(num_classes=args.n_classes)
    else:
        raise ValueError("Model not found.")

    if args.load_model_from_run:
        model.load_state_dict(torch.load(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.load_model_from_run}/checkpoints/weights.pt"))
        # load bn weights as pkles 
        bn_weights = pkl.load(open(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.load_model_from_run}/checkpoints/bn_weights.pkl", "rb"))
        model.bn_weights = bn_weights

    model.to(args.device)

    strategy = MIND(model)

    if args.dataset == 'CIFAR100':
        class_order = list(range(100))
        random.shuffle(class_order)

        train_dataset = CIFAR100(data_path, download=True, train=True)
        test_dataset = CIFAR100(data_path, download=True, train=False)

        '''if args.mode == 3:
            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for i in range(len(old_y)):
                new_y.append(old_y[i])
                new_y.append(old_y[i])
                new_x.append(old_x[i])
                new_x.append(np.rot90(old_x[i], 2))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)'''

        if args.mode == 4:
            # modifico il class order
            class_order_ = []
            for t in range(10):
                for c in range(10):
                    class_order_.append(class_order[t * 10 + c])
                for c in range(10):
                    class_order_.append(class_order[t * 10 + c] + 100)
            class_order = class_order_

            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for i in range(len(old_y)):
                new_y.append(old_y[i])
                new_y.append(old_y[i]+100)
                new_x.append(old_x[i])
                r = old_y[i] % 3
                new_x.append(old_x[i][:, :, [(r+2)%3, (r+1)%3, r]])
                #noise = np.random.normal(0, 10, old_x[i].shape).astype(np.float32)
                #new_x.append(np.clip(noise+old_x[i], 0, 255))
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
                increment=args.classes_per_exp*2,
                class_order=class_order,
                transformations=default_transforms)
        else:
            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp,
                class_order=class_order,
                transformations=default_transforms)

        if args.mode == 4:
            inc = args.classes_per_exp + args.extra_classes
        else:
            inc = args.classes_per_exp

        if args.aug_inf:
            tra = to_tensor
        else:
            tra = to_tensor_and_normalize

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

    elif 'CORE50' in args.dataset :
        data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/MIND/data_64x64/core50_128x128')
        if args.dataset == 'CORE50_CI':
            train_data, test_data = get_all_core50_data(data_path, args.n_experiences, split=0.8)
        else:
            train_data, test_data = get_all_core50_scenario(data_path, split=0.8)
        
        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        strategy.train_scenario = ContinualScenario(
            train_dataset,
            transformations=default_transforms_core50)

        strategy.test_scenario = ContinualScenario(
            test_dataset,
            transformations=to_tensor_and_normalize_core50)

    elif args.dataset == 'TinyImageNet':
        data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/MIND/data_64x64')
        train_data, test_data = get_all_tinyImageNet_data(data_path,args.n_experiences)

        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=default_transforms_TinyImageNet)

        strategy.test_scenario_2 = ClassIncremental(
            test_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=to_tensor_and_normalize_TinyImageNet)

    elif args.dataset == 'Synbols':

        train_data, test_data = get_synbols_data(data_path, n_tasks=args.n_experiences)

        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=default_transforms_Synbols)

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=to_tensor_and_normalize_Synbols)



    print(f"Number of classes: {strategy.train_scenario.nb_classes}.")
    print(f"Number of tasks: {strategy.train_scenario.nb_tasks}.")

    if args.load_model_from_run:
        strategy.pruner.masks = torch.load(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.load_model_from_run}/checkpoints/masks.pt")

    for i, train_taskset in enumerate(strategy.train_scenario):
        if args.packnet_original:
            with torch.no_grad():
                strategy.pruner.dezero(strategy.model)

        strategy.experience_idx = i
        strategy.model.set_output_mask(i, train_taskset.get_classes())
        if args.load_model_from_run:
            model.load_bn_params(strategy.experience_idx)

        # prepare dataset
        strategy.train_taskset, strategy.val_taskset = split_train_val(train_taskset, val_split=args.val_split)
        strategy.train_dataloader = DataLoader(strategy.train_taskset, batch_size=args.bsize, shuffle=True)
        if len(strategy.val_taskset):
            strategy.val_dataloader = DataLoader(strategy.val_taskset, batch_size=args.bsize, shuffle=True)
        else:
            strategy.val_dataloader = DataLoader(strategy.test_scenario_2[i], batch_size=args.bsize, shuffle=True)

        #################### TRAIN ###########################

        # instantiate new model
        if not args.self_distillation:
            if args.model == 'gresnet32':
                strategy.fresh_model = gresnet32(dropout_rate=args.dropout)
            elif args.model == 'gresnet18':
                strategy.fresh_model = gresnet18(num_classes=args.n_classes)
            elif args.model == 'gresnet18mlp':
                strategy.fresh_model = gresnet18mlp(num_classes=args.n_classes)
            else:
                raise ValueError("Model not found.")
        else:
            strategy.fresh_model = deepcopy(strategy.model)
            strategy.distillation = False
            strategy.pruner.set_gating_masks(strategy.fresh_model, strategy.experience_idx, weight_sharing=args.weight_sharing, distillation=strategy.distillation)
        
        strategy.fresh_model.to(args.device)
        strategy.fresh_model.set_output_mask(i, train_taskset.get_classes())

        # instantiate oprimizer
        strategy.train_epochs = args.epochs
        strategy.distillation = False
        strategy.optimizer = torch.optim.AdamW(strategy.fresh_model.parameters(), lr=args.lr, weight_decay=args.wd)
        strategy.scheduler = torch.optim.lr_scheduler.MultiStepLR(strategy.optimizer, milestones=args.scheduler, gamma=0.5, last_epoch=-1, verbose=False)

        print(f'-.-.-.-.-.-. Start training on experience {i+1} - epochs: {strategy.train_epochs} .-.-.-.-.-.')
        strategy.train()

        # Freeze the model for distillation purposes
        strategy.distill_model = freeze_model(deepcopy(strategy.fresh_model))
        strategy.distill_model.to(args.device)

        ########### FINETUNING/DISTILLATION ################
        # selects subset of neurons, prune non selected weights
        if not args.load_model_from_run:
            with torch.no_grad():
                strategy.pruner.prune(strategy.model, strategy.experience_idx, strategy.distill_model, args.self_distillation)


        strategy.train_epochs = args.epochs_distillation
        strategy.distillation = True
        strategy.optimizer = torch.optim.AdamW(strategy.model.parameters(), lr=args.lr_distillation, weight_decay=args.wd_distillation)
        strategy.scheduler = torch.optim.lr_scheduler.MultiStepLR(strategy.optimizer, milestones=args.scheduler_distillation, gamma=0.5, last_epoch=-1, verbose=False)
        print(f"    >>> Start Finetuning epochs: {args.epochs_distillation} <<<")
        strategy.pruner.set_gating_masks(strategy.model, strategy.experience_idx, weight_sharing=args.weight_sharing, distillation=strategy.distillation)
        strategy.train()

        #################### TEST ##########################
        # concatenate pytorch datasets up to the current experience
        with torch.no_grad():
            # write accuracy on the test set
            total_acc = 0
            task_acc = 0
            accuracy_e = 0
            if args.aug_inf:
                for n_aug in range(args.num_aug):
                    if (n_aug % 5) == 0:
                        total_acc, task_acc, accuracy_e, accuracy_taw = test(strategy, strategy.test_scenario[:i + 1], n_aug)
            else:
                total_acc, task_acc, accuracy_e, accuracy_taw = test(strategy, strategy.test_scenario[:i+1], 0)

        #with open(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.run_name}/results/total_acc.csv", "a") as f:
        #    f.write(f"{strategy.experience_idx},{total_acc:.4f}\n")
        #with open(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.run_name}/results/total_acc_taw.csv", "a") as f:
        #    f.write(f"{strategy.experience_idx},{accuracy_taw:.4f}\n")

        # save the model and the masks
        if not args.load_model_from_run:
            os.makedirs(os.path.dirname(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.run_name}/checkpoints/weights.pt"), exist_ok=True)
            torch.save(strategy.model.state_dict(), f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.run_name}/checkpoints/weights.pt")
            torch.save(strategy.pruner.masks, f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.run_name}/checkpoints/masks.pt")
            pkl.dump(strategy.model.bn_weights, open(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{args.run_name}/checkpoints/bn_weights.pkl", "wb"))

    # push results to excel
    #unpublished = True
    #while unpublished:
    #    try:
    #        push_results(args, total_acc, task_acc, accuracy_e, accuracy_taw)
    #        unpublished = False
    #    except:
    #        "Failed to push results, retrying in 1s"
    #        time.sleep(1)

if __name__ == "__main__":

    main()

