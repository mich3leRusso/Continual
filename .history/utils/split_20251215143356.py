import numpy as np 
import torch
from continuum.tasks.base import BaseTaskSet
from continuum.tasks.task_set import TaskSet, TaskType
from typing import Tuple, List

import numpy as np
from continuum.tasks.base import BaseTaskSet
from continuum.tasks.task_set import TaskSet, TaskType
from typing import List, Tuple


def split_train_balanced(
    dataset: BaseTaskSet,
    n_clients: int,
    val_split: float = 0.0,
    seed: int = 1
) -> List[Tuple[TaskSet, TaskSet]]:
    """
    Split a Continuum dataset (e.g. ClassIncremental) into balanced client datasets.
    Each client receives an equal number of samples per class.

    Returns a list of (train_dataset, val_dataset) for each client.
    """

    rng = np.random.RandomState(seed)

    # Get ALL indices
    all_indices = np.arange(len(dataset))

    # Extract labels safely (ClassIncremental-compatible)
    _, all_labels, _ = dataset.get_raw_samples(all_indices)

    classes = np.unique(all_labels)

    # Build class → indices mapping
    class_to_indices = {
        c: all_indices[all_labels == c] for c in classes
    }

    client_splits = []

    for client_id in range(n_clients):
        train_idx, val_idx = [], []

        for c in classes:
            idx = class_to_indices[c].copy()
            rng.shuffle(idx)

            # Split equally across clients
            per_client = len(idx) // n_clients
            start = client_id * per_client
            end = (client_id + 1) * per_client if client_id < n_clients - 1 else len(idx)

            client_class_idx = idx[start:end]

            # Train/val split within client
            n_val = int(len(client_class_idx) * val_split)

            val_idx.extend(client_class_idx[:n_val])
            train_idx.extend(client_class_idx[n_val:])

        # Create TaskSets (H5-safe)
        if dataset.data_type != TaskType.H5:
            x_tr, y_tr, t_tr = dataset.get_raw_samples(train_idx)
            x_va, y_va, t_va = dataset.get_raw_samples(val_idx)
            idx_tr, idx_va = None, None
        else:
            x_tr = x_va = dataset.h5_filename
            y_tr = dataset._y[train_idx] if dataset._y is not None else None
            y_va = dataset._y[val_idx] if dataset._y is not None else None
            t_tr = dataset._t[train_idx] if dataset._t is not None else None
            t_va = dataset._t[val_idx] if dataset._t is not None else None
            idx_tr = dataset.data_indexes[train_idx]
            idx_va = dataset.data_indexes[val_idx]

        train_set = TaskSet(
            x_tr, y_tr, t_tr,
            trsf=dataset.trsf,
            data_type=dataset.data_type,
            data_indexes=idx_tr
        )

        val_set = TaskSet(
            x_va, y_va, t_va,
            trsf=dataset.trsf,
            data_type=dataset.data_type,
            data_indexes=idx_va
        )

        client_splits.append((train_set, val_set))

    return client_splits
