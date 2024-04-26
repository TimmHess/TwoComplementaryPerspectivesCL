from typing import Dict
from copy import deepcopy
import numpy as np
import quadprog
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import BalancedExemplarsBuffer
from avalanche.training.storage_policy import ClassBalancedBuffer

from src.methods.replay_utils import load_buffer_batch, load_entire_buffer

from src.models.utils import freeze_batchnorm_layers, unfreeze_batchnorm_layers


class ERGEMPlugin(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, n_total_memories, num_experiences, total_num_classes, 
                 memory_strength: float,
                 task_incr=False,
                 num_worker=1,
                 lmbda: float=1.0, do_decay_lmbda: bool=False,
                 use_replay_loss: bool=False,
                 use_approx_buffer: bool=True,  # NOTE: for computatinal reasons
                 small_const=1e-3
                ):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.n_total_memories = n_total_memories
        self.task_incr = task_incr
        self.num_worker=num_worker
        self.memory_strength = memory_strength
        self.small_const = small_const

        self.memory: Dict[int, BalancedExemplarsBuffer] = dict()

        self.patterns_per_experience = n_total_memories // num_experiences #int(patterns_per_experience)
        num_classes = total_num_classes // num_experiences
        if self.task_incr:
            num_classes = total_num_classes
        self.storage_policy = ClassBalancedBuffer(
                max_size=self.patterns_per_experience,
                adaptive_size=False,
                total_num_classes=num_classes
            )

        self.G: Tensor = torch.empty(0)
        self.use_approx_buffer = use_approx_buffer
        self.tmp_gradients = None

        # Replay
        self.use_replay_loss = use_replay_loss
        self.lmbda = lmbda
        self.do_decay_lmbda = do_decay_lmbda
        self.replay_batches = None
        self.replay_mbatch = None


    def before_training(self, strategy, **kwargs):
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
        strategy.projection_not_possible = False  # NOTE: only for tracking purposes
        return 


    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("strategy.clock.train_exp_counter:", strategy.clock.train_exp_counter)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.lmbda *= lmbda_decay_factor
            print("New lmbda is:", self.lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        Prepare replay batches
        """
        if strategy.clock.train_exp_counter > 0:
            self.replay_batches = []

            # Get replay batches
            for t in range(strategy.clock.train_exp_counter):
                buffer_ds = load_entire_buffer(self.memory[t], device="cpu")
                
                replay_batch = None
                # Forward pass(es) to obtain reference gradients
                for mb in DataLoader(buffer_ds, num_workers=self.num_worker, 
                                        batch_size=strategy.train_mb_size, pin_memory=False, shuffle=True, drop_last=False):
                    if replay_batch is None:
                        replay_batch = deepcopy(mb)
                        self.replay_batches.append(replay_batch)
                        break

            # Prepare replay_mbatch
            if self.use_replay_loss:
                # Construct replay_mbatch
                for mb in self.replay_batches:
                    if self.replay_mbatch is None:
                        self.replay_mbatch = [mb[0], mb[1], mb[-1]]
                    else:
                        self.replay_mbatch[0] = torch.cat([self.replay_mbatch[0], mb[0]])
                        self.replay_mbatch[1] = torch.cat([self.replay_mbatch[1], mb[1]])
                        self.replay_mbatch[-1] = torch.cat([self.replay_mbatch[-1], mb[-1]])
                
                perm = torch.randperm(len(self.replay_mbatch[0]))
                idx = perm[:strategy.train_mb_size]
                self.replay_mbatch[0] = self.replay_mbatch[0][idx]
                self.replay_mbatch[1] = self.replay_mbatch[1][idx]
                self.replay_mbatch[-1] = self.replay_mbatch[-1][idx]
        return

    
    def before_forward(self, strategy, **kwargs):
        """
        Calculate the loss with respect to the replayed data separately here.
        This enables to weight the losses separately.
        Needs to be done here to prevent gradients being zeroed!
        """
        # Omit if not using replay
        if not self.use_replay_loss:
            return 
        
        # Concatenate replay_batch to current batch
        x_s, y_s, t_s = None, None, None
        if len(self.memory) > 0:  # Only sample if there are stored
            x_s, y_s, t_s = self.replay_mbatch[0].to(strategy.device), \
                            self.replay_mbatch[1].to(strategy.device), \
                            self.replay_mbatch[-1].to(strategy.device)

        # Append to current new-data batch
        if x_s is not None:  # Add
            assert y_s is not None
            assert t_s is not None
            # Assemble minibatch
            strategy.mbatch[0] = torch.cat([strategy.mbatch[0], x_s])
            strategy.mbatch[1] = torch.cat([strategy.mbatch[1], y_s])
            strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], t_s])
        return
    
    
    def before_backward(self, strategy, **kwargs):
        # Current batch (+ replay data) was already forwarded
        # Compute gradients for the loss
        if not self.use_replay_loss:
            loss = strategy.loss.mean()
        else:
            # Apply lambda weighting
            # Loss on current batch
            loss = self.lmbda*strategy.loss[:strategy.train_mb_size].mean()
            if strategy.loss.shape[0] > strategy.train_mb_size:
                # Loss on replay_batch
                loss += (1 - self.lmbda) * strategy.loss[strategy.train_mb_size:].mean()
        strategy.loss = loss
        strategy.loss.backward()

        # Store the resulting gradients
        self.tmp_gradients = {}
        for name, param in strategy.model.named_parameters():
            self.tmp_gradients[name] = param.grad.clone()
        # Reset gradients
        strategy.optimizer.zero_grad()

        # Gather gradients for GEM - after main loss computation
        if strategy.clock.train_exp_counter > 0:
            freeze_batchnorm_layers(strategy.model)
            G = []
            for r_batch in self.replay_batches:
                xref, yref, tid = r_batch[0].to(strategy.device), r_batch[1].to(strategy.device), r_batch[-1].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, tid)
                loss = strategy._criterion(out, yref).mean()
                loss.backward()
            
                G.append(
                        torch.cat(
                            [
                                torch.zeros(p.numel(), device=strategy.device)
                                if 'batchnorm' in n.lower() or p.grad is None
                                else p.grad.flatten().clone()
                                for n, p in strategy.model.named_parameters()
                            ],
                            dim=0,
                        )
                    )
                
            self.G = deepcopy(torch.stack(G))  # (experiences, parameters)
            unfreeze_batchnorm_layers(strategy.model)
            # Reset gradients
            strategy.optimizer.zero_grad()
        return  


    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        for name, param in strategy.model.named_parameters():
                param.grad = self.tmp_gradients[name]

        # Do the GEM gradient projection
        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    torch.zeros(p.numel(), device=strategy.device)
                    if 'batchnorm' in n.lower() or p.grad is None
                    else p.grad.flatten().clone()
                    for n, p in strategy.model.named_parameters()
                ],
                dim=0,
            )
            
            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        strategy.do_project_gradient = to_project  # NOTE: only for tracking purposes
        strategy.projection_not_possible = False  # NOTE: only for tracking purposes
        if to_project:
            print("Projecting gradient")
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
                # ... just continue
                pass

        return
            
        
    def after_training_iteration(self, strategy, **kwargs):
        self.replay_mbatch = None

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """
        t = strategy.clock.train_exp_counter-1
        self.memory[t] = deepcopy(self.storage_policy)
        self.memory[t].update(strategy, **kwargs)

        self.lmbda_weighting = 1


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
        P = 0.5 * (P + P.transpose()) + np.eye(t) * self.small_const #1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
