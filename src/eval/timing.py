import time

from avalanche.evaluation.metrics.timing import TimePluginMetric
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task
from avalanche.evaluation.metric_results import MetricValue, MetricResult


class EpochTime(TimePluginMetric):
    """
    The epoch elapsed time metric.
    This plugin metric only works at training time.

    The elapsed time will be logged after each epoch.
    """

    def __init__(self):
        """
        Creates an instance of the epoch time metric.
        """

        super(EpochTime, self).__init__(reset_at="epoch", emit_at="epoch", mode="train")

        self._prev_time = None
        self.accumulated_time = 0
        return


    def before_forward(self, strategy):
        # Reset timer
        now = time.perf_counter()
        if self._prev_time is None:
            self._prev_time = now
            return
        self._prev_time = time.perf_counter()


    def after_backward(self, strategy):
        # Accumulate time for the batch
        now = time.perf_counter()
        self.accumulated_time += now - self._prev_time


    def _package_result(self, strategy):
        metric_value = self.accumulated_time
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        
        metric_name = get_metric_name(self, strategy,
                                        add_experience=add_exp,
                                        add_task=False)
        # Reset time
        self.accumulated_time = 0
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def __str__(self):
        return "Time_Epoch"