import numpy as np 
from typing import Tuple, List


def split_train_balanced(x_train: np.ndarray, y_train: np.ndarray, n_clients: int = 2, ):
    """
    Split dataset into n_clients datasets with balanced number of samples per class.
    
    :param dataset: A torch dataset, with .x and .y attributes.
    :param n_clients: Number of clients to split the dataset for.
    :param val_split: Fraction of data to use for validation.
    :return: List of tuples (train_dataset, val_dataset) for each client.
    """

    classes = np.unique(y_train)
    
    # Store indexes per class
    class_indexes = {c: np.where(y_train == c)[0] for c in classes}
    
    random_state = np.random.RandomState(seed=1)
    
    client_datasets = []

    for client_id in range(n_clients):
        
        train_idx =  []

        for c in classes:
            # Shuffle class indexes
            idx = class_indexes[c].copy()
            random_state.shuffle(idx)

            # Determine per-client split
            n_samples = len(idx) // n_clients
            start = client_id * n_samples
            end = (client_id + 1) * n_samples if client_id < n_clients - 1 else len(idx)
            client_class_idx = idx[start:end]
            train_idx.extend(client_class_idx)

        x_train[train_idx]
        y_train[train_idx]  
        client_datasets.append()
        


            

    return client_datasets