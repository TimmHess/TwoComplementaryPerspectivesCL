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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset

#from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer, ExperienceBalancedBuffer, ClassBalancedBuffer #ClassTaskBalancedBuffer, 

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ERFullPlugin(SupervisedPlugin):
    """
    Simplified replay strategy for storing the entired dataset.
    From each observed experience we sample a minibatch from memory with the same size as 
    used for training. This way the loss is automatically accumulated correctly. 
    Basically a joint training approixmation with the replay mechanism.
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
            do_decay_lmbda=False
        ):
        """
        # TODO: add docstring
        """
        super().__init__()

        # Memory
        self.n_total_memories = n_total_memories  # Used dynamically

        self.storage_policy = ExperienceBalancedBuffer(
            max_size=self.n_total_memories,
            adaptive_size=False,
            num_experiences=num_experiences
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

        # Losses
        self.replay_criterion = torch.nn.CrossEntropyLoss()
       
        self.replay_loss = 0
        self.iters=0
        return


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
        
        # lmbda_weighting stays 1.0 for the first experience
        if strategy.clock.train_exp_counter > 0:
            self.last_iteration += 1
            if not self.last_iteration > self.lmbda_warmup_steps:
                # Apply linear weighting over the number of warmup steps
                self.lmbda_weighting = self.last_iteration / self.lmbda_warmup_steps
        return


    def before_forward(self, strategy, **kwargs):
        """
        Calculate the loss with respect to the replayed data separately here.
        This enables to weight the losses separately.
        Needs to be done here to prevent gradients being zeroed!
        """

        # Sample memory batch
        x_s, y_s, t_s = None, None, None
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:  # Only sample if there are stored
            x_s, y_s, t_s = self.load_buffer_batch(storage_policy=self.storage_policy, 
                                        strategy=strategy, nb=strategy.train_mb_size)
            # NOTE: this loads a batch for each task

        # Make a copy of the mbatch buffer (because we will need to write the original)
        mbatch_copy = copy.deepcopy(strategy.mbatch)

        # Append to current new-data batch
        if x_s is not None:  # Add
            assert y_s is not None
            assert t_s is not None
            # Assemble minibatch
            mbatch_copy[0] = mbatch_copy[0].to("cpu")
            mbatch_copy[1] = mbatch_copy[1].to("cpu")
            mbatch_copy[-1] = mbatch_copy[-1].to("cpu")
            mbatch_copy[0] = torch.cat([mbatch_copy[0], x_s])
            mbatch_copy[1] = torch.cat([mbatch_copy[1], y_s])
            mbatch_copy[-1] = torch.cat([mbatch_copy[-1], t_s])

            # Shuffle/Permute the mbatch to prevent larger batch-norm issues
            randperm = torch.randperm(mbatch_copy[0].size()[0])
            mbatch_copy[0] = mbatch_copy[0][randperm]
            mbatch_copy[1] = mbatch_copy[1][randperm]
            mbatch_copy[-1] = mbatch_copy[-1][randperm]

            # Forward/Backward all BUT the last batch
            self.iters = (len(mbatch_copy[0]) // strategy.train_mb_size)
            for i in range(self.iters-1):
                # Forward
                bs_idx = i*strategy.train_mb_size
                be_idx = (i+1)*strategy.train_mb_size # TODO: adjust according to default batch size
                strategy.mbatch[0] = mbatch_copy[0][bs_idx:be_idx].to(strategy.device)
                strategy.mbatch[1] = mbatch_copy[1][bs_idx:be_idx].to(strategy.device)
                strategy.mbatch[-1] = mbatch_copy[-1][bs_idx:be_idx].to(strategy.device)
                strategy.mb_output = strategy.forward()
                # Calculte and aggregate Loss
                strategy.loss = strategy.criterion() / self.iters
                
                # Backward
                strategy.loss.backward() 
            
            strategy.loss = 0 # NOTE: need to reset loss here because avalanche does loss+=new_loss which leads to double backward..
        
        # Set the last batch to be then forwarded by the strategy (this is simply to not adjust the strategy itself)
        strategy.mbatch[0] = mbatch_copy[0][-strategy.train_mb_size:].to(strategy.device)
        strategy.mbatch[1] = mbatch_copy[1][-strategy.train_mb_size:].to(strategy.device)
        strategy.mbatch[-1] = mbatch_copy[-1][-strategy.train_mb_size:].to(strategy.device)
        return


    def before_backward(self, strategy, **kwargs):
        # Need to scale last batch as well when replaying
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:
            strategy.loss = strategy.loss / self.iters
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

    def load_buffer_batch(self, storage_policy, strategy, nb=None):
        """
        Wrapper to retrieve a batch of exemplars from the rehearsal memory
        :param nb: Number of memories to return
        :return: input-space tensor, label tensor
        """

        ret_x, ret_y, ret_t = None, None, None
        # Equal amount as batch: Last batch can contain fewer!
        n_exemplars = strategy.train_mb_size if nb is None else nb
        new_dset = self.retrieve_random_buffer_batch(storage_policy, n_exemplars)  # Dataset object

        # Load the actual data
        for sample in DataLoader(new_dset, batch_size=len(new_dset), pin_memory=False, shuffle=True):
            # NOTE: cant move to device because too little VRAM
            x_s, y_s = sample[0], sample[1]
            t_s = sample[-1]  # Task label (for multi-head)

            ret_x = x_s if ret_x is None else torch.cat([ret_x, x_s])
            ret_y = y_s if ret_y is None else torch.cat([ret_y, y_s])
            ret_t = t_s if ret_t is None else torch.cat([ret_t, t_s])

        return ret_x, ret_y, ret_t


    def retrieve_random_buffer_batch(self, storage_policy, n_samples):
        """
        Retrieve a batch of exemplars from the rehearsal memory.
        First sample indices for the available tasks at random, then actually extract from rehearsal memory.
        There is no resampling of exemplars.

        :param n_samples: Number of memories to return
        :return: input-space tensor, label tensor
        """

        # Actually sample
        subsets = []
        for t, _ in storage_policy.buffer_groups.items():
            # Set of idxs
            cnt_idxs = torch.randperm(len(storage_policy.buffer_groups[t].buffer))[:n_samples]
            sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
            sample_idxs = sample_idxs.view(-1)

            # Actual subset
            s = Subset(storage_policy.buffer_groups[t].buffer, sample_idxs.tolist())
            subsets.append(s)
        
        new_dset = ConcatDataset(subsets)
        return new_dset