#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
import torch
import torch.nn.functional as F

from avalanche.evaluation.metric_definitions import GenericPluginMetric, Metric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

class GradProjectionTracker(GenericPluginMetric[float, Accuracy]):
    """
    Evaluation plugin for down-stream tasks.

    Params:
        down_stream_task: The task to evaluate on
        scenario_loader: The scenario loader to use for the down-stream task, i.e. 'get_scenario' from helper.py # NOTE: this can't be importet due to cyclic dependece here..
        num_finetune_epochs: Number of epochs to finetune the model on the down-stream task
        batch_size: Batch size to use for the down-stream task
        num_workers: Number of workers to use for the down-stream task
        skip_initial_eval: If True, the initial evaluation on the down-stream task is skipped   
    """
    def __init__(self, strategy_attr=None):
        self._accuracy = Accuracy() # metric calculation container
        super(GradProjectionTracker, self).__init__(
            self._accuracy, reset_at='iteration', emit_at='iteration', mode='train')
        
        self.strategy_attr = strategy_attr
        self.ret_obj = None
        return

    def __str__(self):
        return self.strategy_attr

    def update(self, strategy):
        try:
            ref_obj = strategy
            ref_obj = getattr(ref_obj, self.strategy_attr)
        except Exception as e:
                return

        if ref_obj is None:
            return

        if isinstance(ref_obj, torch.Tensor):
            ret_obj = ref_obj.detach().clone()
        else:
            ret_obj = torch.tensor(ref_obj)
        self.ret_obj = ret_obj.item()

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        metric_value = 1.0 if self.ret_obj else 0.0

        add_exp = True
        plot_x_position = strategy.clock.train_iterations
        
        metric_name = get_metric_name(self, strategy,
                                        add_experience=add_exp,
                                        add_task=False)
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()

        try: # NOTE: the try-except is a bit hacky, but necessary to avoid crash for initial eval
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return


class FloatAttributeTracker(GenericPluginMetric[float, Accuracy]):
    """
    Evaluation plugin for down-stream tasks.

    Params:
        down_stream_task: The task to evaluate on
        scenario_loader: The scenario loader to use for the down-stream task, i.e. 'get_scenario' from helper.py # NOTE: this can't be importet due to cyclic dependece here..
        num_finetune_epochs: Number of epochs to finetune the model on the down-stream task
        batch_size: Batch size to use for the down-stream task
        num_workers: Number of workers to use for the down-stream task
        skip_initial_eval: If True, the initial evaluation on the down-stream task is skipped   
    """
    def __init__(self, strategy_attr=None):
        self._accuracy = Accuracy() # metric calculation container
        super(FloatAttributeTracker, self).__init__(
            self._accuracy, reset_at='iteration', emit_at='iteration', mode='train')
        
        self.strategy_attr = strategy_attr
        self.ret_obj = None
        return

    def __str__(self):
        return self.strategy_attr

    def update(self, strategy):
        try:
            ref_obj = strategy
            ref_obj = getattr(ref_obj, self.strategy_attr)
        except Exception as e:
                return

        if ref_obj is None:
            return

        if isinstance(ref_obj, torch.Tensor):
            ret_obj = ref_obj.detach().clone()
        else:
            ret_obj = torch.tensor(ref_obj)
        self.ret_obj = ret_obj.item()

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        metric_value = self.ret_obj

        add_exp = True
        plot_x_position = strategy.clock.train_iterations
        
        metric_name = get_metric_name(self, strategy,
                                        add_experience=add_exp,
                                        add_task=False)
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()

        try: # NOTE: the try-except is a bit hacky, but necessary to avoid crash for initial eval
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    