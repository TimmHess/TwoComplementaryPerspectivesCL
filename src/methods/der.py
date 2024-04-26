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

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

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


@torch.no_grad()
def compute_dataset_logits(dataset, model, batch_size, device, num_workers=0):
    was_training = model.training
    model.eval()

    logits = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        out = out.detach().cpu()

        for row in out:
            logits.append(torch.clone(row))

    if was_training:
        model.train()

    return logits


class ClassBalancedBufferWithLogits(BalancedExemplarsBuffer):
    """
    ClassBalancedBuffer that also stores the logits
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = False,
        total_num_classes: Optional[int] = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes is not None and total_num_classes > 0
            ), "When fixed exp mem size, total_num_classes should be > 0."

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes: Set[int] = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data: AvalancheDataset = strategy.experience.dataset

        logits = compute_dataset_logits(
            new_data.eval(),
            strategy.model,
            strategy.train_mb_size,
            strategy.device,
            num_workers=kwargs.get("num_workers", 0),
        )
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(
                    FlatData([logits], discard_elements_not_in_indices=True),
                    name="logits",
                    use_in_getitem=True,
                )
            ],
        )
        # Get sample idxs per class
        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        targets: Sequence[SupportsInt] = getattr(new_data, "targets")
        for idx, target in enumerate(targets):
            # Conversion to int may fix issues when target
            # is a single-element torch.tensor
            target = int(target)
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = new_data_with_logits.subset(c_idxs)
            cl_datasets[c] = subset
        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                # Here it uses underlying dataset
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])


class ClassTaskBalancedBufferWithLogits(BalancedExemplarsBuffer):
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

        logits = compute_dataset_logits(
            new_data.eval(),
            strategy.model,
            strategy.train_mb_size,
            strategy.device,
            num_workers=kwargs.get("num_workers", 0),
        )
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(
                    FlatData([logits], discard_elements_not_in_indices=True),
                    name="logits",
                    use_in_getitem=True,
                )
            ],
        )

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
        
        # NOTE: cl_idxs maps target to all sample indices belonging to that target in the dataset

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = new_data_with_logits.subset(c_idxs)
            cl_datasets[c] = subset

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



class DERPlugin(SupervisedPlugin):
    """
    Implements the DER and the DER++ Strategy,
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    # TODO: fix parameters
    def __init__(
        self,
        mem_size: int = 200,
        total_num_classes=100,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        do_decay_beta: bool = False,
        task_incremental: bool = False,
        num_experiences:int = 1
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
        self.alpha = alpha
        self.beta = beta
        self.do_decay_beta = do_decay_beta

    def before_training(self, strategy, **kwargs):
        if self.batch_size_mem is None:
            self.batch_size_mem = strategy.train_mb_size
        else:
            self.batch_size_mem = self.batch_size_mem
        #raise NotImplementedError("DERPlugin is not meant to be used as a standalone")
        
        # Overwrite the criterion reduction to none
        strategy._criterion.reduction = 'none'  # Overwrite

        # Also overwrite the _make_empty_loss function because it does not work with non reduced losses
        def new_make_empty_loss(self):
            return 0
        strategy._make_empty_loss = new_make_empty_loss.__get__(strategy)

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
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.beta *= lmbda_decay_factor
            print("New lmbda is:", self.beta)
        return


    def before_forward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(strategy.device),
            batch_y.to(strategy.device),
            batch_tid.to(strategy.device),
            batch_logits.to(strategy.device),
        )
        strategy.mbatch[0] = torch.cat((batch_x, strategy.mbatch[0]))
        strategy.mbatch[1] = torch.cat((batch_y, strategy.mbatch[1]))
        strategy.mbatch[2] = torch.cat((batch_tid, strategy.mbatch[2]))
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
        # DER Loss computation
        if self.replay_loader is None:
            strategy.loss = strategy.loss.mean()
        
        else:
            loss_new = strategy.loss[strategy.train_mb_size:].mean()

            if self.do_decay_beta:
                loss_new *= self.beta

            loss_der = self.alpha * F.mse_loss(
                strategy.mb_output[:self.batch_size_mem],
                self.batch_logits,
            )
    
            loss_old = 0
            if self.beta > 0.0:
                loss_old = strategy.loss[:strategy.train_mb_size].mean()
            strategy.loss = loss_new + loss_der + loss_old
        return

    def after_training_exp(self, strategy, **kwargs):
        self.replay_loader = None  # Allow DER to be checkpointed
        self.storage_policy.update(strategy, **kwargs)
        return

    
