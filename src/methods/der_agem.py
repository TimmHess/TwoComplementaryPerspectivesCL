from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    SupportsInt,
    Union,
)
import copy

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.models import avalanche_forward
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.benchmarks.utils.flat_data import FlatData
from avalanche.training.utils import cycle
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.storage_policy import (
    BalancedExemplarsBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.methods.der import ClassBalancedBufferWithLogits, ClassTaskBalancedBufferWithLogits
from src.models.utils import unfreeze_batchnorm_layers, set_batchnorm_layers_to_eval


class DERAGEMPlugin(SupervisedPlugin):
    """
    Implements the DER and the DER++ Strategy,
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    def __init__(
        self,
        mem_size,
        total_num_classes,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        do_decay_beta: bool = False,
        use_agem: bool = False,
        task_incremental: bool = False,
        num_experiences: int = 1,
    ):
        """
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param alpha: float : Hyperparameter weighting the MSE loss
        :param beta: float : Hyperparameter weighting the CE loss,
                             when more than 0, DER++ is used instead of DER
        """
        super().__init__()
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        if task_incremental:
            self.storage_policy = ClassTaskBalancedBufferWithLogits(
                max_size=self.mem_size, 
                adaptive_size=False,
                total_num_classes=total_num_classes*num_experiences
            )
        else:
            self.storage_policy = ClassBalancedBufferWithLogits(
                max_size=self.mem_size,
                adaptive_size=False,
                total_num_classes=total_num_classes
            )
        self.replay_loader = None
        self.replay_mbatch = None
        self.alpha = alpha
        self.beta = beta
        self.do_decay_beta = do_decay_beta

        # AGEM params
        self.use_agem = use_agem
        self.reference_gradients: torch.Tensor = torch.empty(0)   
        self.eps_agem = 1e-7
        self.tmp_gradients = None

    def before_training(self, strategy, **kwargs):
        if self.batch_size_mem is None:
            self.batch_size_mem = strategy.train_mb_size
        else:
            self.batch_size_mem = self.batch_size_mem
        
        # Overwrite the criterion reduction to none
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


    def before_training_exp(self, strategy, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                    num_workers=kwargs.get("num_workers", 0),
                )
            )
        else:
            self.replay_loader = None

        if strategy.clock.train_exp_counter > 0 and self.do_decay_beta:
            beta_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying beta by:", beta_decay_factor)
            self.beta *= beta_decay_factor
            print("New beta is:", self.beta)
        return
    

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if self.replay_loader is not None:
            replay_batch = next(self.replay_loader)
            self.replay_mbatch = copy.deepcopy(replay_batch)
        return
    

    def before_forward(self, strategy, **kwargs):
        # Omit when there is no replay data
        if self.replay_loader is None:
            return

        batch_x, batch_y, batch_tid, batch_logits = self.replay_mbatch
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(strategy.device),
            batch_y.to(strategy.device),
            batch_tid.to(strategy.device),
            batch_logits.to(strategy.device),
        )

        strategy.mbatch[0] = torch.cat((strategy.mbatch[0], batch_x))
        strategy.mbatch[1] = torch.cat((strategy.mbatch[1], batch_y))
        strategy.mbatch[-1] = torch.cat((strategy.mbatch[-1], batch_tid))
        self.batch_logits = batch_logits
        return
    

    def before_backward(self, strategy, **kwargs):
        """
        There are a few difference compared to the autors impl:
            - Joint forward pass vs. 3 forward passes
            - One replay batch vs two replay batches
            - Logits are stored from the non-transformed sample
              after training on task vs instantly on transformed sample
        """

        if self.replay_loader is None:
            strategy.loss = strategy.loss.mean()
            strategy.loss.backward()  # NOTE: Need to backward here because avalanche backward is de-activated
        else:
            # Default loss computation
            loss_old = strategy.loss[strategy.train_mb_size:].mean()
            loss_new = strategy.loss[:strategy.train_mb_size].mean()

            # DER loss computation
            loss_der = self.alpha * F.mse_loss(
                strategy.mb_output[strategy.train_mb_size:],
                self.batch_logits,
            )

            if self.beta > 0.0: # NOTE: DER++
                strategy.loss = self.beta * loss_new + (1-self.beta) * loss_old + loss_der
            else: # NOTE: DER
                strategy.loss = loss_new + loss_der
            # Backward loss
            strategy.loss.backward()
            
            if self.use_agem:
                # Store copy of gradients
                self.tmp_gradients = {}
                for name, param in strategy.model.named_parameters():
                    self.tmp_gradients[name] = param.grad.clone()
                # Reset gradients
                strategy.optimizer.zero_grad()

                # Set batch-norm layers to eval but still calculate gradietns!
                set_batchnorm_layers_to_eval(strategy.model)
                x_s, y_s, t_s = self.replay_mbatch[0], self.replay_mbatch[1], self.replay_mbatch[-1]
                x_s, y_s, t_s = x_s.to(strategy.device), y_s.to(strategy.device), t_s.to(strategy.device)
                # Forward the replay_mbatch through the model
                out = avalanche_forward(strategy.model, x_s, t_s)
                loss = strategy._criterion(out, y_s).mean()
                loss.backward()

                self.reference_gradients = [
                    torch.zeros(p.numel(), device=strategy.device)
                    if p.grad is None
                    else p.grad.flatten().clone()
                    for n, p in strategy.model.named_parameters()
                ]
                self.reference_gradients = copy.deepcopy(torch.cat(self.reference_gradients))
                # Unfreeze the batch-norm
                unfreeze_batchnorm_layers(strategy.model)
                # Reset gradients
                strategy.optimizer.zero_grad()
        return
    

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        # Set the gradeints for the current batch in the model
        if strategy.clock.train_exp_counter > 0:
            for name, param in strategy.model.named_parameters():
                param.grad = self.tmp_gradients[name]

        if self.replay_loader is not None and self.use_agem:
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
                print("Projecting gradient")
                strategy.do_project_gradient = True
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                ) + 1e-7 # Added constant for better numerical stability
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for p in strategy.model.parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                    count += n_param
        return
    

    def after_training_exp(self, strategy, **kwargs):
        self.replay_loader = None  # Allow DER to be checkpointed
        self.storage_policy.update(strategy, **kwargs)
        return

