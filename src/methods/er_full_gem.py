import numpy as np
import random
import copy
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

import quadprog

from typing import TYPE_CHECKING, Dict

from avalanche.models import avalanche_forward
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import BalancedExemplarsBuffer, ExperienceBalancedBuffer
#from src.utils import get_grad_normL2
from src.models.utils import freeze_batchnorm_layers, unfreeze_batchnorm_layers

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from src.methods.replay_utils import load_buffer_batch, load_entire_buffer




class ERFullGEMPlugin(SupervisedPlugin):
    """
    Extending GEM to use its own memory for replay purposes.
    """

    def __init__(self, n_total_memories: int,
                memory_strength: float, 
                lmbda: float,  
                do_decay_lmbda=False,
                use_replay_loss=True):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """
        super().__init__()

        self.use_replay_loss = use_replay_loss

        self.n_total_memories = n_total_memories

        # Memory
        self.memory: Dict[int, BalancedExemplarsBuffer] = dict()
        self.memory_strength = memory_strength
        self.replay_mbatch = None

        # Storage policy (only used as reference to make copies from)
        self.storage_policy = ExperienceBalancedBuffer(
            max_size=self.n_total_memories,
            adaptive_size=False,
            num_experiences=1
        )
    
        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        print(f"[METHOD CONFIG] SUMMARY: ", end='')
        pprint(self.__dict__, indent=2)

        # weighting of replayed loss and current minibatch loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda

        # GEM attributes
        self.G: torch.Tensor = torch.empty(0)
        self.iters = 0

        # Extra
        self.track_loss = 0

    def before_training(self, strategy: 'SupervisedTemplate', **kwargs):
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
        strategy.projection_not_possible = False # NOTE: only for tracking purposes     
        return 


    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            print("\nDecaying lmbda by:", (strategy.clock.train_exp_counter+1) / (strategy.clock.train_exp_counter+2))
            self.lmbda *= (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("New lmbda is:", self.lmbda)
        return
    

    def before_training_iteration(self, strategy, **kwargs):
        """
        Sample memory batch for replay
        """
        if strategy.clock.train_exp_counter > 0:
            self.replay_mbatches = []

            # Sample a batch for every experience
            for t in range(strategy.clock.train_exp_counter):
                # Load mbatch
                replay_batch = load_buffer_batch(self.memory[t], strategy.train_mb_size, device=strategy.device)
                # Store
                self.replay_mbatches.append(replay_batch)

        return


    def before_forward(self, strategy, **kwargs):
        """
        Forward current and replayed data
        """
        if strategy.clock.train_exp_counter > 0:
            assert self.replay_mbatches is not None

            # Concatenate all replay_mbatches
            replay_mbatch = [torch.cat([mb[0] for mb in self.replay_mbatches]),
                             torch.cat([mb[1] for mb in self.replay_mbatches]),
                             torch.cat([mb[-1] for mb in self.replay_mbatches])]
            
            # Make a copy of the mbatch buffer (because we will need to write the original)
            mbatch_copy = copy.deepcopy(strategy.mbatch)
            mbatch_copy[0] = mbatch_copy[0].to("cpu")
            mbatch_copy[1] = mbatch_copy[1].to("cpu")
            mbatch_copy[-1] = mbatch_copy[-1].to("cpu")

            mbatch_copy[0] = torch.cat([mbatch_copy[0], replay_mbatch[0].to("cpu")])
            mbatch_copy[1] = torch.cat([mbatch_copy[1], replay_mbatch[1].to("cpu")])
            mbatch_copy[-1] = torch.cat([mbatch_copy[-1], replay_mbatch[-1].to("cpu")])

            # Permute the minibatch # NOTE: to prevent weird batch_norm statistics
            perm = torch.randperm(mbatch_copy[0].size()[0])
            mbatch_copy[0] = mbatch_copy[0][perm]
            mbatch_copy[1] = mbatch_copy[1][perm]
            mbatch_copy[-1] = mbatch_copy[-1][perm]

            # Accumulate gradients (forward/backward) for all BUT the last batch
            self.iters = (len(mbatch_copy[0]) // strategy.train_mb_size)
            self.track_loss = 0
            for i in range(self.iters-1):
                 # Forward
                bs_idx = i*strategy.train_mb_size
                be_idx = (i+1)*strategy.train_mb_size
                
                strategy.mbatch[0] = mbatch_copy[0][bs_idx:be_idx].to(strategy.device)
                strategy.mbatch[1] = mbatch_copy[1][bs_idx:be_idx].to(strategy.device)
                strategy.mbatch[-1] = mbatch_copy[-1][bs_idx:be_idx].to(strategy.device)
                strategy.mb_output = strategy.forward()
                # Calculte and aggregate Loss
                strategy.loss = strategy.criterion() / self.iters
                self.track_loss += strategy.loss.item()
                # Backward
                strategy.loss.backward() 

            # Reset loss for last batch
            strategy.loss = 0
        
            # Set the last batch to be then forwarded by the strategy (this is simply to not adjust the strategy itself)
            strategy.mbatch[0] = mbatch_copy[0][-strategy.train_mb_size:].to(strategy.device)
            strategy.mbatch[1] = mbatch_copy[1][-strategy.train_mb_size:].to(strategy.device)
            strategy.mbatch[-1] = mbatch_copy[-1][-strategy.train_mb_size:].to(strategy.device)

        return


    def before_backward(self, strategy, **kwargs):
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

            # Calculate reference gradients for GEM
            freeze_batchnorm_layers(strategy.model)
            G = []
            for replay_mbatch in self.replay_mbatches:
                xref, yref, tid = replay_mbatch[0].to(strategy.device), replay_mbatch[1].to(strategy.device), replay_mbatch[-1].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, tid)
                loss = strategy._criterion(out, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            p.grad.flatten().clone()
                            if p.grad is not None
                            else torch.zeros(p.numel(), device=strategy.device)
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )
                    
            self.G = copy.deepcopy(torch.stack(G))
            strategy.loss = torch.Tensor([self.track_loss]) # NOTE: for logging purposes
            strategy.optimizer.zero_grad()
            unfreeze_batchnorm_layers(strategy.model)
        return


    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        # Restore gradients
        if strategy.clock.train_exp_counter > 0:
            for name, param in strategy.model.named_parameters():
                param.grad = self.tmp_gradients[name]


        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    p.grad.flatten().clone()
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )
            
            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        
        strategy.do_project_gradient = False
        strategy.projection_not_possible = False
        if to_project:
            print("Projecting gradient")
            strategy.do_project_gradient = True
            try:
                v_star = self.solve_quadprog(g).to(strategy.device)

                num_pars = 0  # reshape v_star into the parameter matrices
                for p in strategy.model.parameters():
                    curr_pars = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                    num_pars += curr_pars

                assert num_pars == v_star.numel(), "Error in projecting gradient"
            except Exception as e:
                print(e)
                # Set the tracker
                strategy.projection_not_possible = True
                # ... do nothing
                pass
    
    def after_training_exp(self, strategy, **kwargs):
        """
        Update the memory with the current experience.
        """
        t = copy.deepcopy(strategy.clock.train_exp_counter) -1

        # Add a copy of the general storing policy and update it 
        # This way every task has its own, separate, ClassBalancedBuffer
        self.memory[t] = copy.deepcopy(self.storage_policy)
        self.memory[t].update(strategy, **kwargs)
        return

    
    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()