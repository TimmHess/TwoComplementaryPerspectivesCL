from continuum.datasets import CIFAR100
#from continuum.scenarios import ContinualScenario

#from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_generic_benchmark_from_tensor_lists
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import\
    create_generic_benchmark_from_tensor_lists # NOTE: No better way presented yet
#from avalanche.benchmarks.utils import AvalancheDatasetType
import numpy as np
import torch
import copy

#from src.benchmarks.utils import XYDataset
from torch.utils.data import TensorDataset

"""
We can create a instance incremental setting with the coarse labels, i.e. 20 classes. 
Data are labeled with the coarse labels of CIFAR100. However, data are shared between tasks using 
the original label to ensure a domain drift between tasks , e.g., for the coarse label say{aquatic mammals} 
the data go from beavers to dolphins to otters to seals to finally whales in separate tasks.
"""

def get_domain_cifar100_dataset(rootpath, transform=None):
    # Load the dataset from continuum (because the happen to have it prepared)
    train_set = CIFAR100(rootpath,
                        train=True,
                        labels_type="category",
                        task_labels="lifelong")

    test_set = CIFAR100(rootpath,
                        train=False,
                        labels_type="category",
                        task_labels="lifelong")
    # Access the data
    train_data = train_set.get_data() # Returns (imgs(50000), labels(20), task_labels)
    test_data = test_set.get_data() # Returns (imgs(10000), labels(20), task_labels)

    # Merge into dataset
    return TensorDataset(train_data[0], train_data[1], transform=transform), \
           TensorDataset(test_data[0], test_data[1], transform=transform)


def DomainCifar100(rootpath, train_transform=None, eval_transform=None, seed=None):
    # Load the dataset from continuum (because the happen to have it prepared)
    train_set = CIFAR100(rootpath,
                        train=True,
                        labels_type="category",
                        task_labels="lifelong")

    test_set = CIFAR100(rootpath,
                        train=False,
                        labels_type="category",
                        task_labels="lifelong")
    # Access the data
    train_data = train_set.get_data() # Returns (imgs(50000), labels(20), task_labels)
    test_data = test_set.get_data() # Returns (imgs(10000), labels(20), task_labels)

    if seed:
         # Gernerate permutations for classes in tasks
        perm = np.zeros((20,5), dtype=int)
        for class_idx in range(20):
            perm[class_idx] = np.random.permutation(5)
    
    # Prepare TrainSet Split it according to the task labels
    prepared_data = []
    for data in [train_data, test_data]:
        if seed:
            data_copy = copy.deepcopy(data)
            # Apply the permutation to the task labels
            for class_idx in range(20):
                data[2][data[1]==class_idx] = \
                    perm[class_idx][data_copy[2][data[1]==class_idx]]
            
        task_labels = np.unique(data[2])
        x_s = []
        #t_s = []
        for task_label in task_labels:
            imgs = data[0][data[2] == task_label]
            labels = data[1][data[2] == task_label]
            #t = data[2][data[2] == task_label]
            imgs = torch.tensor(imgs)
            imgs = imgs.permute(0,3,1,2)
            labels = torch.tensor(labels)
            #t = torch.tensor(t)
            x_s.append((imgs, labels))
            #t_s.append(torch.Tensor(t))
        #prepared_data.append([x_s, t_s])
        prepared_data.append(x_s)

    # Create the avalanche benchmark
    scenario = create_generic_benchmark_from_tensor_lists(
            train_tensors=prepared_data[0],
            test_tensors=prepared_data[1],
            task_labels=[0,1,2,3,4], # for some resaon this needs to be set in this way by hand... However, we know it in advance...
            complete_test_set_only=False,
            train_transform=train_transform,
            train_target_transform=None,
            eval_transform=eval_transform,
            eval_target_transform=None,
            #dataset_type=AvalancheDatasetType.CLASSIFICATION # NOTE: got removed
    )
    return scenario
