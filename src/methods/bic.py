from collections import defaultdict
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    SupportsInt,
    Union,
)
import copy
import random

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ClassBalancedBuffer,
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models.bic_model import BiasLayer
from avalanche.models import FeatureExtractorModel
from avalanche.models import avalanche_forward

from avalanche.training.templates import SupervisedTemplate

from src.models import BiCClassifier
from src.methods.replay_utils import load_buffer_batch

class BiCPlugin(SupervisedPlugin):
    """
    Bias Correction (BiC) plugin.

    Technique introduced in:
    "Wu, Yue, et al. "Large scale incremental learning." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition. 2019"

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/bic.py
    """

    def __init__(
        self,
        mem_size: int = 2000,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        er_lmbda=0.5,
        do_decay_er_lmbda=False,
        total_num_classes=100,
        val_percentage: float = 0.1,
        T: int = 2,
        stage_2_epochs: int = 200,
        lamb: float = -1,
        lr: float = 0.1,
        num_workers: Union[int, Literal["as_strategy"]] = "as_strategy",
        verbose: bool = False,
        use_agem = True,
    ):
        """
        :param mem_size: replay buffer size.
        :param batch_size: the size of the data batch. If set to `None`, it
            will be set equal to the strategy's batch size.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks. If its value is set to `None`
            (the default value), it will be automatically set equal to the
            data batch size.
        :param task_balanced_dataloader: if True, buffer data loaders will be
                task-balanced, otherwise it will create a single dataloader for
                the buffer samples.
        :param storage_policy: The policy that controls how to add new exemplars
                            in memory
        :param val_percentage: hyperparameter used to set the
                percentage of exemplars in the val set.
        :param T: hyperparameter used to set the temperature
                used in stage 1.
        :param stage_2_epochs: hyperparameter used to set the
                amount of epochs of stage 2.
        :param lamb: hyperparameter used to balance the distilling
                loss and the classification loss.
        :param lr: hyperparameter used as a learning rate for
                the second phase of training.
        :param num_workers: number of workers using during stage 2 data loading.
            Defaults to "as_strategy", which means that the number of workers
            will be the same as the one used by the strategy.
        :param verbose: if True, prints additional info regarding the stage 2 stage
        """

        # Replay (Phase 1)
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.val_percentage = val_percentage
        # if storage_policy is not None:  # Use other storage policy
        #     self.storage_policy = storage_policy
        #     assert storage_policy.max_size == self.mem_size
        # else:  # Default
        #     self.storage_policy = ExperienceBalancedBuffer(
        #         max_size=self.mem_size, adaptive_size=True
        #     )

        # Default to ClassBalancedBuffer
        self.storage_policy = ClassBalancedBuffer(
                #max_size=self.mem_size,
                max_size=int((1-self.val_percentage) * self.mem_size),
                adaptive_size=False,
                total_num_classes=total_num_classes
        )
        
        # Replay hyper-params
        self.er_lmbda = er_lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_er_lmbda = do_decay_er_lmbda

        self.replay_mbatch = None

        # AGEM
        self.use_agem = use_agem
        self.reference_gradients = None
        self.eps_agem = 1e-7

        # Train Bias (Phase 2)
        self.val_percentage = val_percentage
        self.stage_2_epochs = stage_2_epochs
        self.T = T
        self.lamb = lamb
        self.mem_size = mem_size
        self.lr = lr
        self.num_workers: Union[int, Literal["as_strategy"]] = num_workers

        self.seen_classes: Set[int] = set()
        self.class_to_tasks: Dict[int, int] = {}
        self.bias_layer: Optional[BiasLayer] = None
        self.model_old: Optional[Module] = None
        self.val_buffer: Dict[int, ReservoirSamplingBuffer] = {}

        self.is_first_experience: bool = True

        self.verbose: bool = verbose

    # NOTE: the below assert is useless in my case
    # def before_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
    #     assert not isinstance(
    #         strategy.model, MultiTaskModule
    #     ), "BiC only supported for Class Incremetnal Learning (single head)"

    def replace_classifier(self, strategy: "SupervisedTemplate"):
        if not isinstance(strategy.model, FeatureExtractorModel):
            raise ValueError("The strategy's model must be a FeatureExtractorModel")
        if not isinstance(strategy.model.train_classifier, torch.nn.Linear):
            raise ValueError("The strategy's classifier must be a torch.nn.Linear")
        
        device = next(strategy.model.train_classifier.parameters()).device
        bic_classifier = BiCClassifier(
            in_features=strategy.model.train_classifier.in_features,
            num_classes=strategy.model.train_classifier.out_features
        ).to(device)
        
        strategy.model.train_classifier = bic_classifier
        print("\nBiC Plugin replaced the classifier\n")


    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data: AvalancheDataset = strategy.experience.dataset
        task_id = strategy.clock.train_exp_counter

        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        targets: Sequence[SupportsInt] = getattr(new_data, "targets")
        for idx, target in enumerate(targets):
            # Conversion to int may fix issues when target
            # is a single-element torch.tensor
            target = int(target)
            cl_idxs[target].append(idx)

        for c in cl_idxs.keys():
            self.class_to_tasks[c] = task_id

        self.seen_classes.update(cl_idxs.keys())
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        train_data = []
        for class_id in cl_idxs.keys():
            ll = class_to_len[class_id]
            new_data_c = new_data.subset(cl_idxs[class_id][:ll])
            if class_id in self.val_buffer:
                old_buffer_c = self.val_buffer[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.val_buffer[class_id] = new_buffer

            train_data.append(new_data.subset(cl_idxs[class_id][ll:]))

        # resize buffers
        for class_id, class_buf in self.val_buffer.items():
            class_buf.resize(strategy, class_to_len[class_id])

        strategy.experience.dataset = concat_datasets(train_data)

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
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


    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        TODO
        """
        assert strategy.adapted_dataset is not None

        # Adapt the model's classifier if first experience
        if strategy.clock.train_exp_counter == 0:
            self.replace_classifier(strategy)

        # During the distillation phase this layer is not trained and is only
        # used to correct the bias of the classes encountered in the previous experience.
        # It will be unlocked in the bias correction phase.
        if self.bias_layer is not None:
            for param in self.bias_layer.parameters():
                param.requires_grad = False

        if strategy.clock.train_exp_counter == 0: #len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        # batch_size = self.batch_size
        # if batch_size is None:
        #     batch_size = strategy.train_mb_size

        # batch_size_mem = self.batch_size_mem
        # if batch_size_mem is None:
        #     batch_size_mem = strategy.train_mb_size

        # NOTE: We do not alter the startegies dataloader
        # Replay is now handled separately
        # strategy.dataloader = ReplayDataLoader(
        #     strategy.adapted_dataset,
        #     self.storage_policy.buffer,
        #     oversample_small_tasks=True,
        #     batch_size=batch_size,
        #     batch_size_mem=batch_size_mem,
        #     task_balanced_dataloader=self.task_balanced_dataloader,
        #     num_workers=num_workers,
        #     shuffle=shuffle,
        # )
        if strategy.clock.train_exp_counter > 0 and self.do_decay_er_lmbda:
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.er_lmbda *= lmbda_decay_factor
            print("New lmbda is:", self.er_lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        TODO
        """
 
        if strategy.clock.train_exp_counter > 0:
            replay_mbatch = load_buffer_batch(storage_policy=self.storage_policy, 
                                              train_mb_size=strategy.train_mb_size, device=strategy.device)
            self.replay_mbatch = copy.deepcopy(replay_mbatch)


        # Get reference gradients for AGEM
        # if strategy.clock.train_exp_counter > 0:
        #     strategy.model.train()
        #     strategy.optimizer.zero_grad()
        #     mb = load_buffer_batch(storage_policy=self.storage_policy, 
        #                            train_mb_size=strategy.train_mb_size,
        #                            device=strategy.device)

        #     self.replay_mbatch = copy.deepcopy(mb)
            
        #      # Omit if AGEM not in use
        #     if not self.use_agem:
        #         return

        #     xref, yref, tid = mb[0], mb[1], mb[-1]
        #     xref, yref = xref.to(strategy.device), yref.to(strategy.device)

        #     out = avalanche_forward(strategy.model, xref, tid)
        #     loss = strategy._criterion(out, yref).mean()
        #     loss.backward()
        #     # gradient can be None for some head on multi-headed models
        #     self.reference_gradients = [
        #         p.grad.view(-1)
        #         if p.grad is not None
        #         else torch.zeros(p.numel(), device=strategy.device)
        #         for n, p in strategy.model.named_parameters()
        #     ]
        #     self.reference_gradients = torch.cat(self.reference_gradients)
            #strategy.optimizer.zero_grad()
        return

    def before_forward(self, strategy, **kwargs):
        """
        Sample a batch from the memory and append it to the current batch
        """
        # Omit for first experience
        if strategy.clock.train_exp_counter == 0:
            return

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


    def before_backward(self, strategy, **kwargs):
        '''Replay'''

        if strategy.clock.train_exp_counter == 0:
            strategy.loss = strategy.loss.mean()
            strategy.loss.backward()  # NOTE: Need to backward here because avalanche backward is de-activated
        
        else:
            nb_samples = strategy.loss.shape[0]        

            # Get gradients for replayed data
            loss_old = None
            if nb_samples > strategy.train_mb_size:
                loss_old = strategy.loss[strategy.train_mb_size:].mean()
                if self.use_agem:
                    loss_old.backward(retain_graph=True)
                    # Store reference gradients for later AGEM projeciton
                    self.reference_gradients = [
                        torch.zeros(p.numel(), device=strategy.device)
                        #if 'batchnorm' in n.lower() or p.grad is None
                        if p.grad is None
                        else p.grad.flatten().clone()
                        for n, p in strategy.model.named_parameters()
                    ]
                    self.reference_gradients = copy.deepcopy(torch.cat(self.reference_gradients))
                    # Reset gradients
                    strategy.optimizer.zero_grad()
                # Weigh the loss on replayed data for model update
                loss_old *= (1 - self.er_lmbda) 
            
            # Loss on new data
            loss_new = self.er_lmbda * strategy.loss[:strategy.train_mb_size].mean()
            
            # Total loss if loss_old not None
            strategy.loss = loss_new if loss_old is None else loss_new + loss_old

        '''Distillation'''
        if self.model_old is not None:  # That is, from the second experience onwards
            distillation_loss = self.make_distillation_loss(strategy)
            #print("DEBUG bic: distillation loss:", distillation_loss.item())

            # Count the number of already seen classes (i.e., classes from previous experiences)
            initial_classes, previous_classes, current_classes = self._classes_groups(
                strategy
            )

            # Make old_classes and all_classes
            old_clss: Set[int] = set(initial_classes) | set(previous_classes)
            all_clss: Set[int] = old_clss | set(current_classes)

            if self.lamb == -1:
                lamb = len(old_clss) / len(all_clss)
                strategy.loss = (1.0 - lamb) * strategy.loss + lamb * distillation_loss
            else:
                strategy.loss = strategy.loss + self.lamb * distillation_loss

            # Finally, backward loss
            strategy.loss.backward()
        return
    
    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients 
        @ copied avalanche code
        """

        # Omit if AGEM not in use
        if not self.use_agem:
            return

        # Project gradients
        if not self.reference_gradients is None:
            current_gradients = [
                torch.zeros(p.numel(), device=strategy.device)
                if p.grad is None
                else p.grad.flatten().clone()
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients)

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

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(
                            grad_proj[count : count + n_param].view_as(p)
                        )
                    count += n_param
        return
    

    def after_training_iteration(self, strategy, **kwargs):
        """
        When using continual eval, we need to trigger the BiC updates every iteration, and 
        and apply the bias layer accordingly.
        """
        # strategy.tracking_collector.is_tracking_iteration
        if hasattr(strategy, "tracking_collector"):
            if strategy.tracking_collector.is_tracking_iteration\
                and not strategy.clock.train_exp_counter == 0:
                num_workers = (
                    int(kwargs.get("num_workers", 0))
                    if self.num_workers == "as_strategy"
                    else self.num_workers
                )
                persistent_workers = (
                    False if num_workers == 0 else kwargs.get("persistent_workers", False)
                )

                self.bias_correction_step(
                    strategy,
                    persistent_workers=persistent_workers,
                    num_workers=num_workers,
                )
                

    def after_training_exp(self, strategy, **kwargs):
        '''Replay'''
        # Update memory
        self.storage_policy.update(strategy, **kwargs)
        self.er_lmbda_weighting = 1

        '''BiC'''
        self.is_first_experience = False

        self.model_old = None
        self.model_old = copy.deepcopy(strategy.model)
        self.model_old.eval()
        for param in self.model_old.parameters():
            param.requires_grad = False

        task_id = strategy.clock.train_exp_counter

        self.storage_policy.update(strategy, **kwargs)

        if task_id > 0:
            num_workers = (
                int(kwargs.get("num_workers", 0))
                if self.num_workers == "as_strategy"
                else self.num_workers
            )
            persistent_workers = (
                False if num_workers == 0 else kwargs.get("persistent_workers", False)
            )

            self.bias_correction_step(
                strategy,
                persistent_workers=persistent_workers,
                num_workers=num_workers,
            )

    def bias_forward(self, input_data: Tensor) -> Tensor:
        if self.bias_layer is None:
            return input_data

        return self.bias_layer(input_data)

    def cross_entropy(self, new_outputs, old_outputs):
        """Calculates cross-entropy with temperature scaling"""
        dis_logits_soft = torch.nn.functional.softmax(old_outputs / 2, dim=0)
        loss_distill = torch.nn.functional.cross_entropy(
            new_outputs / 2, dis_logits_soft
        )
        return loss_distill

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        max_size = int(self.val_percentage * self.mem_size)
        lengths = [max_size // num_groups for _ in range(num_groups)]
        # distribute remaining size among experiences.
        rem = max_size - sum(lengths)
        for i in range(rem):
            lengths[i] += 1

        return lengths

    def make_distillation_loss(self, strategy):
        assert self.model_old is not None
        initial_classes, previous_classes, current_classes = self._classes_groups(
            strategy
        )

        # Forward current minibatch through the old model
        with torch.no_grad():
            out_old: Tensor = self.model_old(strategy.mb_x)

        if len(initial_classes) == 0:
            # We are in the second experience, no need to correct the bias
            # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L561
            pass
        else:
            # We are in the third experience or later
            # bias_forward will apply the bias correction to the output of the old model for the classes
            # found in previous_classes (bias correction is not applied to initial_classes or current_classes)!
            # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L564
            assert self.bias_layer is not None
            assert set(self.model_old.train_classifier.bias_layer.clss.tolist()) == set(previous_classes)
            with torch.no_grad():
                out_old = self.model_old.train_classifier.bias_layer(out_old)

        # To compute the distillation loss, we need the output of the new model
        # without the bias correction. During train, the output of the new model
        # does not undergo bias correction, so we can use mb_output directly.
        out_new: Tensor = strategy.mb_output

        # Union of initial_classes and previous_classes: needed to select the logits of all the old classes
        old_clss: List[int] = sorted(set(initial_classes) | set(previous_classes))

        # Distillation loss on the logits of the old classes
        return self.cross_entropy(out_new[:, old_clss], out_old[:, old_clss])

    def bias_correction_step(
        self,
        strategy: SupervisedTemplate,
        persistent_workers: bool = False,
        num_workers: int = 0,
    ):
        
        # --- Prepare the models ---
        # Freeze the base model, only train the new bias layer
        strategy.model.eval()


        # Nullify any previous bias_layer
        if isinstance(strategy.model.train_classifier, BiCClassifier):
            strategy.model.train_classifier.bias_layer = None

        # Create the bias layer of the current experience
        targets = getattr(strategy.adapted_dataset, "targets")

        self.bias_layer = BiasLayer(targets.uniques)
        self.bias_layer.to(strategy.device)
        self.bias_layer.train()
        for param in self.bias_layer.parameters():
            param.requires_grad = True

        bic_optimizer = torch.optim.SGD(
            self.bias_layer.parameters(), lr=self.lr, momentum=0.9
        )

        # Typing note: verbose here is actually correct
        # The PyTorch type stubs for MultiStepLR are broken in some versions
        scheduler = MultiStepLR(
            bic_optimizer, milestones=[10, 20, 25], gamma=0.1, verbose=False
        )  # type: ignore

        # --- Prepare the dataloader for the validation set ---
        list_subsets: List[AvalancheDataset] = []
        for _, class_buf in self.val_buffer.items():
            list_subsets.append(class_buf.buffer)

        stage_set = concat_datasets(list_subsets)
        stage_loader = DataLoader(
            stage_set,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # Loop epochs
        for e in range(self.stage_2_epochs):
            total, t_acc, t_loss = 0, 0, 0
            for inputs in stage_loader:
                x = inputs[0].to(strategy.device)
                y_real = inputs[1].to(strategy.device)

                with torch.no_grad():
                    outputs = strategy.model(x)
                    #outputs = strategy.model.get_features(x)

                outputs = self.bias_layer(outputs)

                loss = torch.nn.functional.cross_entropy(outputs, y_real)

                _, preds = torch.max(outputs, 1)
                t_acc += torch.sum(preds == y_real.data)
                t_loss += loss.item() * x.size(0)
                total += x.size(0)

                # Hand-made L2 loss
                # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L636
                loss += 0.1 * ((self.bias_layer.beta.sum() ** 2) / 2)

                bic_optimizer.zero_grad()
                loss.backward()
                bic_optimizer.step()

            scheduler.step()
            if self.verbose and (self.stage_2_epochs // 4) > 0:
                if (e + 1) % (self.stage_2_epochs // 4) == 0:
                    print(
                        "| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |".format(
                            e + 1, t_loss / total, 100 * t_acc / total
                        )
                    )

        # Reset the backbone to training mode
        strategy.model.train()

        # Freeze the bias layer
        self.bias_layer.eval()
        for param in self.bias_layer.parameters():
            param.requires_grad = False

        if self.verbose:
            print(
                "Bias correction done: alpha={}, beta={}".format(
                    self.bias_layer.alpha.item(), self.bias_layer.beta.item()
                )
            )
        
        # Copy bias layer to startegy's classifier after first experience is done
        if not isinstance(strategy.model.train_classifier, BiCClassifier):
            raise ValueError("The strategy's classifier must be a BiCClassifier")
        if strategy.clock.train_exp_counter > 0:
            strategy.model.train_classifier.bias_layer = self.bias_layer
            

    def _classes_groups(self, strategy: SupervisedTemplate):
        current_experience: int = strategy.experience.current_experience
        # Split between
        # - "initial" classes: seen between in experiences [0, current_experience-2]
        # - "previous" classes: seen in current_experience-1
        # - "current" classes: seen in current_experience

        # "initial" classes
        initial_classes: Set[
            int
        ] = set()  # pre_initial_cl in the original implementation
        previous_classes: Set[int] = set()  # pre_new_cl in the original implementation
        current_classes: Set[int] = set()  # new_cl in the original implementation
        # Note: pre_initial_cl + pre_new_cl is "initial_cl" in the original implementation

        for cls, exp_id in self.class_to_tasks.items():
            assert exp_id >= 0
            assert exp_id <= current_experience

            if exp_id < current_experience - 1:
                initial_classes.add(cls)
            elif exp_id == current_experience - 1:
                previous_classes.add(cls)
            else:
                current_classes.add(cls)

        return (
            sorted(initial_classes),
            sorted(previous_classes),
            sorted(current_classes),
        )


__all__ = [
    "BiCPlugin",
]
