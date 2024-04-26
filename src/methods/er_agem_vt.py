import numpy as np
import random
import copy
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from typing import TYPE_CHECKING

from avalanche.models import avalanche_forward
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
#from src.utils import get_grad_normL2

from src.models.utils import freeze_batchnorm_layers, unfreeze_batchnorm_layers

import time

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from src.methods.replay import ClassTaskBalancedBuffer
from src.methods.replay_utils import load_buffer_batch

class ERAGEMPlugin(SupervisedPlugin):
    """
    Rehearsal Revealed: replay plugin.
    Implements two modes: Classic Experience Replay (ER) and Experience Replay with Ridge Aversion (ERaverse).
    """
    store_criteria = ['rnd']

    def __init__(self, n_total_memories: int, sample_size: int,
                lmbda: float, lmbda_warmup_steps=0, do_decay_lmbda=False,
                task_incremental=False, total_num_classes=100, num_experiences=1,
                use_replay_loss=True):
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
                total_num_classes=total_num_classes*num_experiences
            )
        else:
            self.storage_policy = ClassBalancedBuffer(
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes
            )
        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        pprint(self.__dict__, indent=2)

        # weighting of replayed loss and current minibatch loss
        self.use_replay_loss = use_replay_loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda
        
        self.nb_new_samples = None
        self.replay_mbatch = None

        # AGEM parameters
        self.tmp_gradients = None
        self.reference_gradients = None
        self.eps_agem = 1e-7


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


    def before_training_exp(self, strategy: 'SupervisedTemplate', **kwargs):
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.lmbda *= lmbda_decay_factor
            print("New lmbda is:", self.lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        Sample memory batch for replay
        """
        self.nb_new_samples = strategy.mb_x.shape[0]
        
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:  # Only sample if there are stored            
            # Get and store replay_mbatch for later use
            replay_mbatch = load_buffer_batch(storage_policy=self.storage_policy, 
                                              train_mb_size=strategy.train_mb_size, device=strategy.device)
            self.replay_mbatch = copy.deepcopy(replay_mbatch)
        return


    def before_forward(self, strategy, **kwargs):
        """
        Concatenate replay_mbatch to strategy.mbatch if using replay loss
        """
        # Omit concatinating replay_mbatch to strategy.mbatch when not using replay loss
        if not self.use_replay_loss:
            return
        
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
        # If is first experience
        if strategy.clock.train_exp_counter == 0:
            strategy.loss = strategy.loss.mean()
            strategy.loss.backward()  # NOTE: Need to backward here because avalanche backward is de-activated
        else:
            # Backward loss on replay_mbatch to get reference gradients for AGEM
            loss_old = strategy.loss[strategy.train_mb_size:].mean()
            loss_old.backward(retain_graph=True)

            self.reference_gradients = parameters_to_vector(strategy.model.parameters()).clone()
    
            # Weight the gradients
            # Multiply every gradient in the model by (1-self.lmbda)
            for n, p in strategy.model.named_parameters():
                if p.grad is not None:
                    p.grad *= (1-self.lmbda)
            
            # Backward weighted loss of current batch
            loss_curr = self.lmbda * strategy.loss[:strategy.train_mb_size].mean()
            loss_curr.backward()
        return
    

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients 
        @ copied avalanche code
        """
        # Set the gradeints for the current batch in the model
        if strategy.clock.train_exp_counter > 0:
            assert self.reference_gradients is not None

            current_gradients = parameters_to_vector(strategy.model.parameters()).clone()

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

                vector_to_parameters(grad_proj, strategy.model.parameters())    
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
        self.replay_mbatch = None
        return