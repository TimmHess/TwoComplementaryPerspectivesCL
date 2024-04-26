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
from avalanche.training.storage_policy import ExemplarsBuffer, ExperienceBalancedBuffer, ClassBalancedBuffer

from src.models.utils import freeze_batchnorm_layers, unfreeze_batchnorm_layers, set_batchnorm_layers_to_eval


if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ERFullAGEMPlugin(SupervisedPlugin):
    """
    Simplified replay strategy for storing the entired dataset.
    From each observed experience we sample a minibatch from memory with the same size as 
    used for training. This way the loss is automatically accumulated correctly. 
    Basically a joint training approixmation with the replay mechanism.
    """

    def __init__(self, 
            n_total_memories, 
            lmbda,   
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
        pprint(self.__dict__, indent=2)

        # weighting of replayed loss and current minibatch loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda
       
        self.iters=0

        # AGEM
        self.tmp_gradients = None
        self.reference_gradients = None
        self.eps_agem = 1e-7

    def before_training(self, strategy, **kwargs):
        """ 
        Omit reduction in criterion to be able to 
        separate losses from buffer and batch
        """    
        # NOTE: Need to nullify the default backward pass of the strategy
        def backward(self):
            return
        strategy.backward = backward.__get__(strategy)

        # Write tracker to strategy
        strategy.do_project_gradient = False  # NOTE: only for tracking purposes
        strategy.magnitude_curr_grads = 0
        strategy.magnitude_ref_grads = 0
        strategy.grad_cosine_similarity = 0
        
        return 

    def before_training_exp(self, strategy, **kwargs):
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

        # Get reference gradients for AGEM
        if strategy.clock.train_exp_counter > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            
            mb = self.load_buffer_batch(storage_policy=self.storage_policy, 
                                        strategy=strategy, nb=strategy.train_mb_size)
            # NOTE: this loads a batch for each task

            self.replay_mbatch = copy.deepcopy(mb)  # Store for later replay
        return


    def before_forward(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            self.forward_replay_buffer(strategy, with_curr_batch=True, with_last_batch=False)
        return


    def before_backward(self, strategy, **kwargs):
        # Need to scale last batch as well when replaying
        if strategy.clock.train_exp_counter == 0:
            strategy.loss.backward()
        else:
            # Backward the final minibatch loss
            strategy.loss = strategy.loss / self.iters
            strategy.loss.backward()
    
            # Store gradients from forward pass
            self.tmp_gradients = {}
            for name, param in strategy.model.named_parameters():
                self.tmp_gradients[name] = param.grad.clone()
    
            # Reset gradients
            strategy.optimizer.zero_grad()

            # Calculate reference gradients for AGEM
            freeze_batchnorm_layers(strategy.model)
            self.forward_replay_buffer(strategy, with_curr_batch=False, with_last_batch=True)
            unfreeze_batchnorm_layers(strategy.model)

            # Store reference gradients for AGEM
            self.reference_gradients = [
                    torch.zeros(p.numel(), device=strategy.device)
                    if p.grad is None
                    else p.grad.flatten().clone()
                    for n, p in strategy.model.named_parameters()
                ]
            self.reference_gradients = copy.deepcopy(torch.cat(self.reference_gradients))
            # Reset gradients
            strategy.optimizer.zero_grad()

        return


    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients 
        @ copied avalanche code
        """

        # Restore gradients
        if strategy.clock.train_exp_counter > 0:
            for name, param in strategy.model.named_parameters():
                param.grad = self.tmp_gradients[name]


        if not self.reference_gradients is None:
            current_gradients_list = [
                torch.zeros(p.numel(), device=strategy.device)
                if p.grad is None
                else p.grad.flatten().clone()
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients) 
            strategy.do_project_gradient = False
            strategy.magnitude_curr_grads = torch.norm(current_gradients)
            strategy.magnitude_ref_grads = torch.norm(self.reference_gradients)
            strategy.grad_cosine_similarity = torch.nn.functional.cosine_similarity(self.reference_gradients, current_gradients, dim=0)
            if dotg < 0:
                strategy.do_project_gradient = True
                alpha2 = dotg / (
                    torch.dot(self.reference_gradients, self.reference_gradients) 
                    + self.eps_agem
                )
                
                grad_proj = (
                    current_gradients - self.reference_gradients * alpha2
                )

                num_pars = 0  # reshape v_star into the parameter matrices
                for p in strategy.model.parameters():
                    curr_pars = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[num_pars : num_pars + curr_pars].view(p.size()))
                    num_pars += curr_pars
        return


    def after_training_exp(self, strategy, **kwargs):
        """ Update memories."""
        self.storage_policy.update(strategy, **kwargs)  # Storage policy: Store the new exemplars in this experience
        self.reset()
        return
    

    def forward_replay_buffer(self, strategy, with_curr_batch=True, with_last_batch=False):
        assert self.replay_mbatch is not None
        # Sample memory batch
        x_s, y_s, t_s = self.replay_mbatch[0], self.replay_mbatch[1], self.replay_mbatch[-1]
        x_s, y_s, t_s = x_s.to("cpu"), y_s.to("cpu"), t_s.to("cpu")

        # Make a copy of the mbatch buffer (because we will need to write the original)
        mbatch_copy = copy.deepcopy(strategy.mbatch)

        # Merge current and replayed data
        if x_s is not None:  # Add
            assert y_s is not None
            assert t_s is not None
            
            # Assemble minibatch
            if with_curr_batch:
                mbatch_copy[0] = mbatch_copy[0].to("cpu")
                mbatch_copy[1] = mbatch_copy[1].to("cpu")
                mbatch_copy[-1] = mbatch_copy[-1].to("cpu")
                mbatch_copy[0] = torch.cat([mbatch_copy[0], x_s])
                mbatch_copy[1] = torch.cat([mbatch_copy[1], y_s])
                mbatch_copy[-1] = torch.cat([mbatch_copy[-1], t_s])
            else:
                mbatch_copy[0] = x_s
                mbatch_copy[1] = y_s
                mbatch_copy[-1] = t_s

            # Shuffle/Permute the mbatch to prevent larger batch-norm issues
            randperm = torch.randperm(mbatch_copy[0].size()[0])
            mbatch_copy[0] = mbatch_copy[0][randperm]
            mbatch_copy[1] = mbatch_copy[1][randperm]
            mbatch_copy[-1] = mbatch_copy[-1][randperm]

            # Accumulate gradients (forward/backward) for all BUT the last batch
            self.iters = (len(mbatch_copy[0]) // strategy.train_mb_size)

            track_loss = 0
            rng = self.iters if with_last_batch else self.iters -1 # NOTE: iters-1 to have the last batch forwarded by avalanche
            for i in range(rng):
                # Forward
                bs_idx = i*strategy.train_mb_size
                be_idx = (i+1)*strategy.train_mb_size 
                strategy.mbatch[0] = mbatch_copy[0][bs_idx:be_idx].to(strategy.device)
                strategy.mbatch[1] = mbatch_copy[1][bs_idx:be_idx].to(strategy.device)
                strategy.mbatch[-1] = mbatch_copy[-1][bs_idx:be_idx].to(strategy.device)
                strategy.mb_output = strategy.forward()
                # Calculte and aggregate Loss
                strategy.loss = strategy.criterion() / self.iters
                track_loss += strategy.loss.item()
                # Backward
                strategy.loss.backward() 
            
            # Reset loss for last batch
            if not with_last_batch:
                strategy.loss = 0 # NOTE: need to reset loss here because avalanche does loss+=new_loss which leads to double backward..
            else:
                strategy.loss = torch.Tensor([track_loss]).to(strategy.device) 
        
        # Set the last batch to be then forwarded by the strategy (this is simply to not adjust the strategy itself)
        if not with_last_batch:
            strategy.mbatch[0] = mbatch_copy[0][-strategy.train_mb_size:].to(strategy.device)
            strategy.mbatch[1] = mbatch_copy[1][-strategy.train_mb_size:].to(strategy.device)
            strategy.mbatch[-1] = mbatch_copy[-1][-strategy.train_mb_size:].to(strategy.device)
    
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