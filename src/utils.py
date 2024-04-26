from avalanche.core import BaseSGDPlugin

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


def args_to_tensorboard(writer, args):
    txt = ""
    for arg in sorted(vars(args)):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"
    writer.add_text('command_line_parameters', txt, 0)
    return



class IterationsInsteadOfEpochs(BaseSGDPlugin):
    """Stop training based on number of iterations instead of epochs."""

    def __init__(self, max_iterations: int):
        super().__init__()
        self.max_iterations = max_iterations -1 # -1 because we start at 0

    def before_training_exp(self, strategy: 'SupervisedTemplate', **kwargs):
        if self.max_iterations == 0:
            strategy.stop_training()
        return super().before_training_exp(strategy, **kwargs)

    def after_training_iteration(self, strategy, **kwargs):
        if strategy.clock.train_exp_iterations == self.max_iterations:
            print(f"Stopping training, reached max iterations: {self.max_iterations}")
            strategy.stop_training()