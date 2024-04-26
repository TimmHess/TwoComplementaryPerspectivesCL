#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Adapted by Timm Hess (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import random
import copy
from pprint import pprint
from typing import TYPE_CHECKING, Optional, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset

from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.benchmarks.utils.flat_data import FlatData
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer, \
    BalancedExemplarsBuffer, ReservoirSamplingBuffer

from avalanche.benchmarks.utils.data import AvalancheDataset

from src.methods.replay_utils import load_buffer_batch, retrieve_random_buffer_batch, compute_dataset_logits

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ClassTaskBalancedBuffer(BalancedExemplarsBuffer):
    """ Stores samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each
        class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True,
                 total_num_classes: int = None):
        """
        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        self.task_shift = 1000
        

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        # Access the current experience dataset
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}

        # Check and get the task_label
        assert len(np.unique(new_data.targets_task_labels)) == 1, "Only one task label is supported"
        task_label = np.unique(new_data.targets_task_labels)[0]
        
        for idx, target in enumerate(new_data.targets):
            target = int(target+task_label*self.task_shift) # NOTE: 1000 should be bigger than max number of tasks!
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheDataset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(num_groups=self.total_num_classes)#len(self.seen_classes)
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy,
                                                class_to_len[class_id])



class ACE_CE_Loss(nn.Module):
    """
    Masked version of CrossEntropyLoss.
    """
    def __init__(self, device):
        super(ACE_CE_Loss, self).__init__()

        self.seen_so_far = torch.zeros(0, dtype=torch.int).to(device) # basically an empty tensor
        return

    def forward(self, logits, labels):
        present = labels.unique()

        mask = torch.zeros_like(logits).fill_(-1e9)
        mask[:, present] = logits[:, present] # add the logits for the currently observed labels
        
        if len(self.seen_so_far) > 0: # if there are seen classes, add them as well (this is for replay buffer loss)
            mask[:, (self.seen_so_far.max()+1):] = logits[:, (self.seen_so_far.max()+1):] # add the logits for the unseen labels
        
        logits = mask
        return F.cross_entropy(logits, labels)


class ERPlugin(SupervisedPlugin):
    """
    Rehearsal Revealed: replay plugin.
    Implements two modes: Classic Experience Replay (ER) and Experience Replay with Ridge Aversion (ERaverse).
    """
    store_criteria = ['rnd']

    def __init__(self, 
            n_total_memories, 
            lmbda, 
            device, 
            replay_batch_handling='separate', # NOTE: alterantive is 'combined'
            task_incremental=False, 
            domain_incremental=False,
            lmbda_warmup_steps=0, 
            total_num_classes=100, 
            num_experiences=1,
            do_decay_lmbda=False, 
            ace_ce_loss=False
        ):
        """
        # TODO: add docstring
        """
        super().__init__()

        # Memory
        self.n_total_memories = n_total_memories  # Used dynamically
        # a Dict<task_id, Dataset>
        if task_incremental:
            self.storage_policy = ClassTaskBalancedBuffer(  # Samples to store in memory
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes
            )
        elif domain_incremental:
            self.storage_policy = ClassTaskBalancedBuffer(  # Samples to store in memory
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes*num_experiences # NOTE: because classes are repeated...
            )
        else:
            self.storage_policy = ClassBalancedBuffer(
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes
            )
        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        print(f"[METHOD CONFIG] SUMMARY: ", end='')
        print(f"[METHOD CONFIG] replay_batch_handling={replay_batch_handling} ")
        pprint(self.__dict__, indent=2)

        # device
        self.device = device

        # weighting of replayed loss and current minibatch loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda
        self.lmbda_warmup_steps = lmbda_warmup_steps
        self.do_exp_based_lmbda_weighting = False
        self.last_iteration = 0

        # replay batch handling
        self.replay_batch_handling = replay_batch_handling
        self.nb_new_samples = None
        self.replay_mbatch = None

        # Losses
        self.replay_criterion = torch.nn.CrossEntropyLoss()
        self.use_ace_ce_loss = ace_ce_loss
        if self.use_ace_ce_loss:
            self.replay_criterion = ACE_CE_Loss(self.device)
            self.ace_ce_loss = ACE_CE_Loss(self.device)
        self.replay_loss = 0


    def before_training(self, strategy: 'SupervisedTemplate', **kwargs):
        """ 
        Omit reduction in criterion to be able to 
        separate losses from buffer and batch
        """
        strategy._criterion.reduction = 'none'  # Overwrite

        # Also overwrite the _make_empty_loss function because it does not work with non reduced losses
        def new_make_empty_loss(self):
            return 0
        
        strategy._make_empty_loss = new_make_empty_loss.__get__(strategy)
        #instance.some_method = new_method.__get__(instance)

        return super().before_training(strategy, **kwargs)


    def before_training_exp(self, strategy: 'SupervisedTemplate', **kwargs):
        if self.replay_batch_handling == 'combined':
            return
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.lmbda *= lmbda_decay_factor
            print("New lmbda is:", self.lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        Adjust the lmbda weighting according to lmbda warmup settings
        """
        self.nb_new_samples = strategy.mb_x.shape[0]
        
        # DEBUG
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:  # Only sample if there are stored
            strategy.model.train()
            strategy.optimizer.zero_grad()
            replay_mbatch = load_buffer_batch(storage_policy=self.storage_policy, 
                                              train_mb_size=strategy.train_mb_size, device=strategy.device)
            self.replay_mbatch = copy.deepcopy(replay_mbatch)
        return


    def before_forward(self, strategy, **kwargs):
        """
        Calculate the loss with respect to the replayed data separately here.
        This enables to weight the losses separately.
        Needs to be done here to prevent gradients being zeroed!
        """

        # Sample memory batch
        x_s, y_s, t_s = None, None, None
            
        if self.replay_mbatch is not None:
            x_s, y_s, t_s = self.replay_mbatch[0], self.replay_mbatch[1], self.replay_mbatch[-1]
            x_s, y_s, t_s = x_s.to(strategy.device), y_s.to(strategy.device), t_s.to(strategy.device)

        # Append to current new-data batch
        if x_s is not None:  # Add
            assert y_s is not None
            assert t_s is not None
            # Assemble minibatch
            strategy.mbatch[0] = torch.cat([strategy.mbatch[0], x_s])
            strategy.mbatch[1] = torch.cat([strategy.mbatch[1], y_s])
            strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], t_s])
        return
    

    def before_backward(self, strategy: 'SupervisedTemplate', **kwargs):
        # Disentangle losses
        nb_samples = strategy.loss.shape[0]        

        #print("DEBUG: replay - loss new", strategy.loss[:self.nb_new_samples].mean(), self.lmbda)
        loss_new = self.lmbda * strategy.loss[:self.nb_new_samples].mean()
        loss = loss_new

        # Mem loss
        if nb_samples > self.nb_new_samples:
            loss_reg = (1 - self.lmbda) * strategy.loss[self.nb_new_samples:].mean()
            loss = loss_new + loss_reg  

        # Writeback loss to strategy   
        strategy.loss = loss      
        return
    

    def after_training_exp(self, strategy, **kwargs):
        """ Update memories."""
        self.storage_policy.update(strategy, **kwargs)  # Storage policy: Store the new exemplars in this experience
        self.reset()
        return

    def reset(self):
        """
        Reset internal variables after each experience
        """
        self.last_iteration = 0
        self.lmbda_weighting = 1
        return