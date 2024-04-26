import numpy as np
import random
import copy
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from typing import TYPE_CHECKING, Optional, List

from avalanche.models import avalanche_forward
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
#from src.utils import get_grad_normL2

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class AGEMPlugin(SupervisedPlugin):
    """
    Extending GEM to use its own memory for replay purposes.
    """

    def __init__(self, n_total_memories: int, sample_size: int,
                lmbda: float, lmbda_warmup_steps=0, do_decay_lmbda=False):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """
        super().__init__()

        self.n_total_memories = n_total_memories

        # a Dict<task_id, Dataset>
        self.storage_policy = ClassBalancedBuffer(  # Samples to store in memory
            max_size=self.n_total_memories,
            adaptive_size=True,
        )
        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        print(f"[METHOD CONFIG] SUMMARY: ", end='')
        pprint(self.__dict__, indent=2)

        # weighting of replayed loss and current minibatch loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda
        self.lmbda_warmup_steps = lmbda_warmup_steps
        self.do_exp_based_lmbda_weighting = False
        self.last_iteration = 0

        # AGEM parameters
        self.reference_gradients = None
        self.sample_size = sample_size
        self.eps_agem = 1e-7
        

    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            print("\nDecaying lmbda by:", (strategy.clock.train_exp_counter+1) / (strategy.clock.train_exp_counter+2))
            self.lmbda *= (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("New lmbda is:", self.lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        1) Compute reference gradient on memory sample 
        2) Adjust the lmbda weighting according to lmbda warmup settings
        """
        # Get reference gradients for AGEM
        if strategy.clock.train_exp_counter > 0 and self.sample_size > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.load_buffer_batch(storage_policy=self.storage_policy,
                    strategy=strategy, nb=self.sample_size)
            
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            loss = strategy._criterion(out, yref)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            self.reference_gradients = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()

        # lmbda_weighting stays 1.0 for the first experience
        if strategy.clock.train_exp_counter > 0:
            self.last_iteration += 1
            #print("before_iteration:", self.last_iteration, self.lmbda_warmup_steps)
            if not self.last_iteration > self.lmbda_warmup_steps:
                # Apply linear weighting over the number of warmup steps
                self.lmbda_weighting = self.last_iteration / self.lmbda_warmup_steps
                #print("lmbda_weighting is:", self.lmbda_weighting)
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
        return

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients 
        @ copied avalanche code
        """
        if not self.reference_gradients is None:
            current_gradients = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / (
                    torch.dot(self.reference_gradients, self.reference_gradients) 
                    + self.eps_agem
                )
                
                grad_proj = (
                    current_gradients - self.reference_gradients * alpha2
                )

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(
                            grad_proj[count : count + n_param].view_as(p)
                        )
                    count += n_param
        return
    
    def after_training_exp(self, strategy, **kwargs):
        """
        Update the memory with the current experience.
        """
        self.storage_policy.update(strategy, **kwargs)
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
        for sample in DataLoader(new_dset, batch_size=len(new_dset), pin_memory=True, shuffle=False):
            x_s, y_s = sample[0].to(strategy.device), sample[1].to(strategy.device)
            t_s = sample[-1].to(strategy.device)  # Task label (for multi-head)
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
        assert n_samples > 0, "Need positive nb of samples to retrieve!"

        # Determine how many mem-samples available
        q_total_cnt = 0  # Total samples
        free_q = {}  # idxs of which ones are free in mem queue
        tasks = []
        for t, ex_buffer in storage_policy.buffer_groups.items():
            mem_cnt = len(ex_buffer.buffer)  # Mem cnt
            free_q[t] = list(range(0, mem_cnt))  # Free samples
            q_total_cnt += len(free_q[t])  # Total free samples
            tasks.append(t)

        # Randomly sample how many samples to idx per class
        free_tasks = copy.deepcopy(tasks)
        tot_sample_cnt = 0
        sample_cnt = {c: 0 for c in tasks}  # How many sampled already
        max_samples = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
        while tot_sample_cnt < max_samples:
            t_idx = random.randrange(len(free_tasks))
            t = free_tasks[t_idx]  # Sample a task

            if sample_cnt[t] >= len(storage_policy.buffer_group(t)):  # No more memories to sample
                free_tasks.remove(t)
                continue
            sample_cnt[t] += 1
            tot_sample_cnt += 1

        # Actually sample
        s_cnt = 0
        subsets = []
        for t, t_cnt in sample_cnt.items():
            if t_cnt > 0:
                # Set of idxs
                cnt_idxs = torch.randperm(len(storage_policy.buffer_group(t)))[:t_cnt]
                sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
                sample_idxs = sample_idxs.view(-1)

                # Actual subset
                s = Subset(storage_policy.buffer_group(t), sample_idxs.tolist())
                subsets.append(s)
                s_cnt += t_cnt
        assert s_cnt == tot_sample_cnt == max_samples
        new_dset = ConcatDataset(subsets)

        return new_dset