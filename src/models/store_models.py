import torch

from pathlib import Path

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class StoreModelsPlugin(SupervisedPlugin):
    def __init__(self, model_name, model_store_path):
        super().__init__()

        self.model_name = model_name
        self.model_store_path = model_store_path
        return

    def store_model(self, strategy):
        # Store model to path
        dir_path = str(self.model_store_path) + "/model_weights/"
        file_name =  self.model_name + "_" + str(strategy.clock.train_exp_counter) + ".pth"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        torch.save(strategy.model.state_dict(), dir_path+file_name)
        print("\nStoring model to path: ", (dir_path+file_name))
        return

    def after_training_exp(self, strategy, **kwargs):
        self.store_model(strategy)
        return 