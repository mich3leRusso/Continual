import numpy as np 
import torch
from continuum.tasks.base import BaseTaskSet
from continuum.tasks.task_set import TaskSet, TaskType
from typing import Tuple, List


def split_train_balanced(dataset: BaseTaskSet, n_clients: int = 2, val_split: float = 0.0) -> List[Tuple[TaskSet, TaskSet]]:
    """
    Split dataset into n_clients datasets with balanced number of samples per class.
    
    :param dataset: A torch dataset, with .x and .y attributes.
    :param n_clients: Number of clients to split the dataset for.
    :param val_split: Fraction of data to use for validation.
    :return: List of tuples (train_dataset, val_dataset) for each client.
    """
    
 
    x, y , t= dataset.dataset


    classes = np.unique(y)
    
    # Store indexes per class
    class_indexes = {c: np.where(y == c)[0] for c in classes}
    
    random_state = np.random.RandomState(seed=1)
    
    client_datasets = []

    for client_id in range(n_clients):
       # train_idx, val_idx = [], []

        for c in classes:
            # Shuffle class indexes
            idx = class_indexes[c].copy()
            #random_state.shuffle(idx)

            # Determine per-client split
            n_samples = len(idx) // n_clients
            start = client_id * n_samples
            end = (client_id + 1) * n_samples if client_id < n_clients - 1 else len(idx)
            client_class_idx = idx[start:end]

            # Split into train/val
            val_count = int(len(client_class_idx) * val_split)
            val_idx=client_class_idx[:val_count]
            train_idx=client_class_idx[val_count:]

            x_train=x[train_idx]
            x_val= x[val_idx]

            y_train=y[train_idx]   
            y_val=y[val_idx] 


            t_train=t[train_idx]   
            t_val=t[val_idx] 
            

    return client_datasets