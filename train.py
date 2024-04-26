from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

import torch
from pathlib import Path
import random
import numpy
import os
import copy

from datetime import datetime
from distutils.util import strtobool

import avalanche as avl
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# CUSTOM
from src.models import get_model #, get_model_summary
from src.utils import args_to_tensorboard

import helper
from cmd_parser import get_arg_parser

'''
Init device
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEBUG stupid cuda init
cuda_init_tensor = torch.rand(1).to(device)
del cuda_init_tensor

'''
Init Argparser
'''
parser = get_arg_parser()
# Load args from commandline
args = parser.parse_args()

# Override args from benchmark yaml file
helper.overwrite_args_with_config(args, args.benchmark_config)

# Override args from strategy yaml file
helper.overwrite_args_with_config(args, args.strategy_config)

'''
Check if GPU is required
'''
print("args.requires_gpu:", args.requires_gpu)
if args.requires_gpu:
    assert torch.cuda.device_count() > 0, "No GPU found!"  


"""
Setup num threads
"""
if args.restrict_num_threads:
    torch.set_num_threads(args.restrict_num_threads)
if args.restrict_num_interop_threads:
    torch.set_num_interop_threads(args.restrict_num_interop_threads)


"""
Setups seeds
"""
def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if args.seed is None:
    args.seed = random.randint(0, 100000)
    print("No seed specified - using random seed:", args.seed)
# Set the seed
set_seed(args.seed, deterministic=True)


'''
Setup results directories
'''
args.now = datetime.now().strftime("%Y.%m.%d-%H:%M")
#optim_str = "no_reset_optim" if args.reset_optim_each_exp else "reset_optim" 
args.setupname = '_'.join([args.exp_name, args.strategy, args.backbone, str(args.lmbda), args.scenario, args.optim , f"e={args.epochs[0]}", args.now]) # removed optim_str after args.optim
args.results_path = Path(os.path.join(args.save_path, args.setupname)).resolve()
args.eval_results_dir = args.results_path / 'results_summary'  # Eval results

for path in [args.results_path, args.eval_results_dir]:
    path.mkdir(parents=True, exist_ok=True)

'''
Create Scenario
'''
scenario, data_transforms, input_size, initial_out_features = helper\
    .get_scenario(
        args=args, 
        scenario_name=args.scenario, 
        dset_rootpath=args.dset_rootpath, 
        num_experiences=args.num_experiences,
        use_data_aug=args.use_data_aug,
        seed=args.seed
    )

'''
Create Logger
'''
loggers = []
# Tensorboard
tb_log_dir = os.path.join(args.results_path)  # Group all runs
tb_logger = TensorboardLogger(tb_log_dir=tb_log_dir)
loggers.append(tb_logger)  # log to Tensorboard
print(f"[Tensorboard] tb_log_dir={tb_log_dir}")
# Terminal
print_logger = TextLogger() 
if args.disable_pbar:
    print_logger = InteractiveLogger()  # print to stdout
loggers.append(print_logger)

'''
Init Evaluation
'''
metrics = helper.get_metrics(scenario, args, data_transforms=data_transforms)
eval_plugin = EvaluationPlugin(*metrics, loggers=loggers) # benchmark=scenario <- deprectated

'''
Init Model
'''
model = get_model(args=args, n_classes=scenario.n_classes, 
                  input_size=input_size, 
                  initial_out_features=initial_out_features,
                  backbone_weights=args.backbone_weights,
                  model_weights=args.pretrained_weights)

'''
Init Strategy
'''
strategy = helper.get_strategy(args, model, eval_plugin, scenario, 
                    device, plugins=[], data_transforms=data_transforms)

'''
Store args to tensorboard
'''
args_to_tensorboard(tb_logger.writer, args)


'''
Train Loop
'''
print('Starting experiment...')
for experience in scenario.train_stream:
#for i, experience in enumerate(scenario.train_stream):
    # TRAIN
    print(f"\n{'-' * 40} TRAIN {'-' * 40}")
    print(f"Start training on experience {experience.current_experience}")
    print("exp_len:", len(experience.dataset))
    strategy.train(
                experience, 
                num_workers=args.num_workers, 
                eval_streams=None)
    print(f"End training on experience {experience.current_experience}")

    # EVAL ALL TASKS (ON TASK TRANSITION)
    print(f"\n{'=' * 40} EVAL {'=' * 40}")
    print(f'Standard Continual Learning eval on entire test set on task transition.')
    task_results_file = args.eval_results_dir / f'seed={args.seed}' / f'task{experience.current_experience}_results.pt'
    task_results_file.parent.mkdir(parents=True, exist_ok=True)
    res = strategy.eval(scenario.test_stream)  # Gathered by EvalLogger

    # Store eval task results
    task_metrics = dict(strategy.evaluator.all_metric_results)
    torch.save(task_metrics, task_results_file)
    print(f"[FILE:TASK-RESULTS]: {task_results_file}")

    if args.terminate_after_exp:
        if strategy.training_exp_counter >= args.terminate_after_exp:
            print("\n\nTERMINATING TRAINING AFTER EXPERIENCE", strategy.training_exp_counter)
            break

# HACK: necessary in python 3.8 to kill all tensorboard writers (otherwise hangs endlessly)
tb_logger.writer.flush()
for lggr in loggers:
    del lggr

print("Experiment finished.")