from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from copy import deepcopy
import shutil
from typing import List

import numpy as np
import PIL.Image as Image

import torch
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms

from avalanche.benchmarks import SplitMNIST, RotatedMNIST
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.evaluation.metrics import ExperienceForgetting, StreamForgetting, accuracy_metrics, loss_metrics, \
    StreamConfusionMatrix, timing_metrics
from avalanche.evaluation.continual_eval import ContinualEvaluationPhasePlugin
from avalanche.evaluation.continual_eval_metrics import TaskTrackingLossPluginMetric,\
        TaskTrackingAccuracyPluginMetric,TaskTrackingMINAccuracyPluginMetric,\
        TaskAveragingPluginMetric, WCACCPluginMetric
from src.eval.grad_projection import GradProjectionTracker, FloatAttributeTracker

from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised import Naive, Cumulative

from src.benchmarks.miniimagenet_benchmark import SplitMiniImageNet
from src.benchmarks.domain_cifar100 import DomainCifar100

from avalanche.training.plugins import GEMPlugin, AGEMPlugin

from avalanche.training.plugins import BiCPlugin as BiCPluginAvalanche
from avalanche.training.plugins import LRSchedulerPlugin

from src.methods import ERFullPlugin, ERPlugin, ERFullGEMPlugin, ERFullAGEMPlugin, \
                        ERAGEMPlugin, ERGEMPlugin, \
                        BiCPlugin, DERPlugin, DERAGEMPlugin
from src.methods.er_gem_v2 import ERGEMPlugin as ERGEMPluginV2
from src.methods.er_agem_v2 import ERAGEMPlugin as ERAGEMPluginV2
from src.methods.er_agem_vt import ERAGEMPlugin as ERAGEMPluginVT
from src.methods.er_vt import ERAGEMPlugin as ERPluginVT

from src.utils import IterationsInsteadOfEpochs
from src.models.store_models import StoreModelsPlugin
from src.eval.timing import EpochTime


def cut_string(s):
    index = s.find('_e=')
    if index != -1:
        s = s[:index]
    return s

def check_overwrite_directory(s):
    # Split the string into directories
    print("")
    print(s)
    s = cut_string(s)
    print("\ns:", s)
    dirs = s.split('/')
    
    # Get the penultimate directory
    penultimate_dir = '/'.join(dirs[:-1])
    
    # Get the last directory
    last_dir = dirs[-1]
    print("penultimate_dir:", penultimate_dir)
    print("last_dir:", last_dir)
    
    # Check if the last directory exists in the penultimate directory
    dir_match = None
    if os.path.exists(penultimate_dir):
        for dirpath, dirnames, filenames in os.walk(penultimate_dir):
            if dir_match is not None:
                break
            print("dirpath:", dirpath)
            print("dirnames:", dirnames)
            for dirname in dirnames:    
                if last_dir in dirname:
                    dir_match = os.path.join(dirpath, dirname)
                    break
   
    if dir_match is None:
        return False, dir_match, penultimate_dir

    # Check if dir_match contains a broken.txt file
    if dir_match is not None:
        if os.path.exists(os.path.join(dir_match, "broken.txt")):
            return False, dir_match, penultimate_dir
    
    return True, dir_match, penultimate_dir

def mark_results_dir(results_dir):
    """Mark results_dir as complete by deleting the broken.txt file, if existing. Otherwise create the broken.txt file."""
    if os.path.exists(results_dir):
        if os.path.exists(os.path.join(results_dir, "broken.txt")):
            os.remove(os.path.join(results_dir, "broken.txt"))
        else:
            with open(os.path.join(results_dir, "broken.txt"), "w") as f:
                f.write("")
    else:
        raise ValueError(f"Results directory {results_dir} does not exist!")


def get_transforms(data_aug, 
                   input_size, 
                   norm_mean=(0.0, 0.0, 0.0), 
                   norm_std=(1.0, 1.0, 1.0),
                   use_to_pil=False,
                   use_corruption=False):
    """
    Single place in codebase where data transforms are defined.
    
    Return: List of transforms for train and test
    """
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize(size=(input_size[1], input_size[2])) #interpolation=transforms.InterpolationMode.NEAREST

    crop_flip = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2]),
                                    scale=(0.1 if input_size[0]>=64 else 0.2, 1.),
                                    #interpolation=transforms.InterpolationMode.BICUBIC #interpolation=Image.BICUBIC
                                    ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    sim_clr = transforms.Compose([
        #transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2]),
                                    scale=(0.1 if input_size[0]>=64 else 0.2, 1.),
                                    #interpolation=transforms.InterpolationMode.BICUBIC #interpolation=Image.BICUBIC
                                    ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, 
                                    contrast=0.4,
                                    saturation=0.2, 
                                    hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=input_size[0]//20*2+1, 
                    sigma=(0.1, 2.0)
                )], 
            p=0.5 if input_size[0]>32 else 0.0),
        #Solarization(p=0.0),
    ])
    rand_crop_aug = transforms.Compose([
        transforms.RandomCrop(size=(input_size[1], input_size[2]), padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ])
    autoaug_cifar10 = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10)
    ])
    autoaug_imgnet = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
    ])
    
    rand_grayscale = transforms.Compose([transforms.RandomGrayscale(p=0.2)]) # TODO
    rand_gauss_blur = transforms.Compose([
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=input_size[0]//20*2+1, sigma=(0.1, 2.0))
                    ], p=0.5 if input_size[0]>32 else 0.0
                    )
                ])

    normalize = transforms.Normalize(norm_mean, norm_std)
    to_tensor = transforms.ToTensor()

    train_transforms = []
    test_transforms = []
    # Optional addition of ToPILImage transform
    if use_to_pil:
        train_transforms.append(to_pil)
        test_transforms.append(to_pil)
    # Default addtion of Resize transform
    train_transforms.append(resize)
    test_transforms.append(resize)
    
    print("selected data augmentation:", data_aug)
    if "simclr" in data_aug:
        train_transforms.append(sim_clr)
    elif "crop_flip" in data_aug:
        train_transforms.append(crop_flip)
    elif "rand_crop" in data_aug:
        train_transforms.append(rand_crop_aug)
        print("Using random crop augmentation")
    elif "auto_cifar10" in data_aug:
        train_transforms.append(autoaug_cifar10)
    elif "auto_imgnet" in data_aug:
        train_transforms.append(autoaug_imgnet)
    
    
    # Default addition of Normalize and ToTensor transforms
    train_transforms.extend([to_tensor, normalize])
    test_transforms.extend([to_tensor, normalize])

    # Finally, compose everything into a single transform
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def get_scenario(args, scenario_name, dset_rootpath,
                num_experiences=None, use_data_aug=None, seed=42):
    print(f"\n[SCENARIO] {scenario_name}, Task Incr = {args.task_incr}")

    # Check for 'none' string #NOTE: this will happen when using default roots for downstream sets
    if not dset_rootpath is None:
        if dset_rootpath.lower() == "none":
            dset_rootpath = None

    # Prepare general transforms
    train_transform = None
    test_transform = None
    data_transforms = dict()

    if scenario_name in ["smnist", "pmnist", "rotmnist"]:
        input_size = (1, 28, 28)
    elif scenario_name in ["cifar10", "cifar100", "domain_cifar100", "digits", "svhn",\
                           "corrupted_cifar100"]:
        input_size = (3, 32, 32)
    elif scenario_name in ["cub200", "tinyimgnet"]:
        input_size = (3, 64, 64) # likewise (3,128,128) or even (3,256,256)
    elif scenario_name in ["minidomainnet", "domain_camelyon17"]:
        input_size = (3, 96, 96)
    elif scenario_name in ["miniimgnet", "corrupted_miniimgnet"]:
        input_size = (3, 84, 84)
    elif scenario_name in ["flowers102"]:
        input_size = (3, 64, 64)
    elif scenario_name in ["imagenet_A, imagenet_R"]:
        input_size = (3, 224, 224)
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

    if not args.overwrite_input_size is None:
        input_size = (input_size[0], args.overwrite_input_size[0], args.overwrite_input_size[1])

    norm_mean = None
    norm_stddev = None
    if args.overwrite_mean and args.overwrite_stddev:
        norm_mean = tuple(args.overwrite_mean)
        norm_stddev = tuple(args.overwrite_stddev)
        print("Overwriting mean and stddev with", norm_mean, norm_stddev)

    n_classes = None
    fixed_class_order = args.fixed_class_order

    # Prepare datasets/scenarios
    if scenario_name == 'smnist': 
        n_classes = 10
        n_experiences = 5

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=True,
                                            norm_mean=(0.1307,), 
                                            norm_std=(0.3081,)
        )
        
        scenario = SplitMNIST(
                    n_experiences=n_experiences, 
                    return_task_id=args.task_incr, 
                    seed=seed,
                    fixed_class_order=[i for i in range(n_classes)], 
                    dataset_root=dset_rootpath,
                    train_transform=train_transform,
                    eval_transform=test_transform
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes // n_experiences  # For Multi-Head

    elif scenario_name == "rotmnist":
        n_classes = 10
        n_experiences = 20
        
        if not args.rotations_list is None:
            n_experiences = len(args.rotations_list)
        else:
            raise ValueError("Rotations list is None! Please provide a list of rotations for RotatedMNIST!")
        print("Using rotations list:", args.rotations_list)
        print("n_experiences:", n_experiences)

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=True,
                                            norm_mean=(0.1307,),  
                                            norm_std=(0.3081,)            
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy  

        scenario = RotatedMNIST(dataset_root=dset_rootpath,
                                n_experiences=n_experiences,
                                return_task_id=args.task_incr,
                                rotations_list=args.rotations_list,
                                train_transform=altered_train_transform,
                                eval_transform=test_transform,
                                seed=args.seed
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes #// n_experiences # because its domain incremental!

    # CIFAR100
    elif scenario_name == 'cifar100':
        n_classes = 100
        n_experiences = 10

        if not fixed_class_order:
            fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            assert not args.fixed_class_order, "Cannot use random class ordering with fixed class ordering!"
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        fixed_class_order = [int(x) for x in fixed_class_order]

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=False,
                                            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
                                            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225) # (0.2675, 0.2565, 0.2761)
        )
        print("train transform:")
        print(train_transform)
        altered_train_transform = train_transform
        
        scenario = SplitCIFAR100(
                    n_experiences=n_experiences, 
                    return_task_id=args.task_incr, 
                    seed=seed,
                    fixed_class_order=fixed_class_order,
                    train_transform=altered_train_transform,
                    eval_transform=test_transform,
                    dataset_root=dset_rootpath
        )
        
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes // n_experiences

    # Domain Cifar100
    elif scenario_name == 'domain_cifar100':
        n_classes = 20
        n_experiences = 5

        assert num_experiences is None, "domain cifar100 will always give 5 experiences! Remove overwriting of experiences!"

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=True, # NOTE: needs to be True here because its comming from a TensorDataset
                                            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  
                                            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225) 
        )
        altered_train_transform = train_transform 

        scenario = DomainCifar100(
                    rootpath=dset_rootpath,
                    train_transform=altered_train_transform,
                    eval_transform=test_transform,
                    seed=args.seed if args.use_rand_class_ordering else None #NOTE: None is default task order
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes 


    # MiniImageNet
    elif scenario_name == 'miniimgnet':
        n_classes = 100
        n_experiences = 20

        fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            print("Using random class ordering for MiniImageNet...")
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=True,
                                            norm_mean=norm_mean if norm_mean else (0.4914, 0.4822, 0.4465), 
                                            norm_std=norm_stddev if norm_stddev else (0.2023, 0.1994, 0.2010)
        )

        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy
  
        scenario = SplitMiniImageNet(
                    dset_rootpath,
                    n_experiences=n_experiences, 
                    return_task_id=args.task_incr, 
                    seed=seed, 
                    fixed_class_order=fixed_class_order, 
                    preprocessed=True,
                    train_transform=altered_train_transform, 
                    test_transform=test_transform 
        )
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes // n_experiences  # For Multi-Head



    # Get initial_out_features
    initial_out_features = scenario.n_classes
    if args.task_incr:
        initial_out_features = n_classes // n_experiences

    # Cutoff if applicable
    scenario.train_stream = scenario.train_stream[: args.partial_num_tasks]
    scenario.test_stream = scenario.test_stream[: args.partial_num_tasks]

    # Pack transforms #NOTE: this is necessary because I am unable to retrieve the transforms from the scenario object (or avalanche dataset)
    data_transforms["train"] = train_transform
    data_transforms["eval"] = test_transform
    print(f"Scenario = {scenario_name}")

    return scenario, data_transforms, input_size, initial_out_features


def get_continual_evaluation_plugins(args, scenario):
    """Plugins for per-iteration evaluation in Avalanche."""
    assert args.eval_periodicity >= 1, "Need positive "

    if args.eval_with_test_data:
        args.evalstream_during_training = scenario.test_stream  # Entire test stream
    else:
        args.evalstream_during_training = scenario.train_stream  # Entire train stream
    print(f"Evaluating on stream (eval={args.eval_with_test_data}): {args.evalstream_during_training}")

    # Metrics
    loss_tracking = TaskTrackingLossPluginMetric()
    
    # Expensive metrics
    gradnorm_tracking = None
    # if args.track_gradnorm:
    #     gradnorm_tracking = TaskTrackingGradnormPluginMetric() # if args.track_gradnorm else None  # Memory+compute expensive
    # featdrift_tracking = None
    # if args.track_featdrift:
    #     featdrift_tracking = TaskTrackingFeatureDriftPluginMetric() # if args.track_features else None  # Memory expensive

    # Acc derived plugins
    acc_tracking = TaskTrackingAccuracyPluginMetric()
    #lca = TrackingLCAPluginMetric()

    acc_min = TaskTrackingMINAccuracyPluginMetric()
    acc_min_avg = TaskAveragingPluginMetric(acc_min)
    wc_acc_avg = WCACCPluginMetric(acc_min)

    # wforg_10 = WindowedForgettingPluginMetric(window_size=10)
    # wforg_10_avg = TaskAveragingPluginMetric(wforg_10)
    # wforg_100 = WindowedForgettingPluginMetric(window_size=100)
    # wforg_100_avg = TaskAveragingPluginMetric(wforg_100)

    # wplast_10 = WindowedPlasticityPluginMetric(window_size=10)
    # wplast_10_avg = TaskAveragingPluginMetric(wplast_10)
    # wplast_100 = WindowedPlasticityPluginMetric(window_size=100)
    # wplast_100_avg = TaskAveragingPluginMetric(wplast_100)

    tracking_plugins = [
        loss_tracking, gradnorm_tracking, acc_tracking, #featdrift_tracking
        #lca,  # LCA from A-GEM (is always avged)
        acc_min, acc_min_avg, wc_acc_avg,  # min-acc/worst-case accuracy
        # wforg_10, wforg_10_avg,  # Per-task metric, than avging metric
        # wforg_100, wforg_100_avg,  # Per-task metric, than avging metric
        # wplast_10, wplast_10_avg,  # Per-task metric, than avging metric
        # wplast_100, wplast_100_avg,  # Per-task metric, than avging metric
    ]
    tracking_plugins = [p for p in tracking_plugins if p is not None]

    trackerphase_plugin = ContinualEvaluationPhasePlugin(tracking_plugins=tracking_plugins,
                                                         max_task_subset_size=args.eval_task_subset_size,
                                                         eval_stream=args.evalstream_during_training,
                                                         eval_max_iterations=args.eval_max_iterations,
                                                         mb_update_freq=args.eval_periodicity,
                                                         num_workers=args.num_workers,
    )
    return [trackerphase_plugin, *tracking_plugins]

def get_metrics(scenario, args, data_transforms):
    """Metrics are calculated efficiently as running avgs."""

    # Pass plugins, but stat_collector must be called first
    minibatch_tracker_plugins = []

    # Stats on external tracking stream
    if args.enable_continual_eval:
        tracking_plugins = get_continual_evaluation_plugins(args, scenario)
        minibatch_tracker_plugins.extend(tracking_plugins)

    # Current minibatch stats per class
    # if args.track_class_stats:
    #     for y in range(scenario.n_classes):
    #         minibatch_tracker_plugins.extend([
    #             # Loss components
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_numerator_c{y}"]),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_denominator_c{y}"]),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_c{y}"]),

    #             # Prototypes
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protodelta_weight_c{y}']),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protonorm_weight_c{y}']),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protodelta_bias_c{y}']),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protonorm_bias_c{y}']),
    #         ])

    # METRICS FOR STRATEGIES (Will only track if available for method)
    if args.track_extra_stats:
        minibatch_tracker_plugins.extend([
            GradProjectionTracker(strategy_attr="do_project_gradient"),
            GradProjectionTracker(strategy_attr="projection_not_possible"),
            FloatAttributeTracker(strategy_attr="magnitude_curr_grads"),
            FloatAttributeTracker(strategy_attr="magnitude_ref_grads"),
            FloatAttributeTracker(strategy_attr="grad_cosine_similarity")
        #     StrategyAttributeTrackerPlugin(strategy_attr=["loss_new"]),
        #     StrategyAttributeTrackerPlugin(strategy_attr=["loss_reg"]),
        #     StrategyAttributeTrackerPlugin(strategy_attr=["gradnorm_stab"]),
        #     StrategyAttributeTrackerPlugin(strategy_attr=["avg_gradnorm_G"]),
        ])

    metrics = [
        accuracy_metrics(minibatch=False, experience=True, stream=True), 
        loss_metrics(minibatch=True, experience=True, stream=False),
        #ExperienceForgetting(),  # Test only
        #StreamForgetting(),  # Test only
        #StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=True),

        # CONTINUAL EVAL
        *minibatch_tracker_plugins,

        # LOG OTHER STATS
        #timing_metrics(epoch=True, experience=False),
        # cpu_usage_metrics(experience=True),
        # DiskUsageMonitor(),
        # MinibatchMaxRAM(),
        # GpuUsageMonitor(0),
    ]

    if args.track_timing:
        print("Tracking Timing...")
        #metrics.append(timing_metrics(epoch=True, experience=False))
        metrics.append(EpochTime())

    # Early Stopping
    # if args.early_stopping:
    #     print("Adding early stopping plugin")
    #     assert scenario.valid_stream is not None, "Early stopping requires a validation stream"
    #     if args.iterations_per_task:
    #         metrics.append(EarlyStoppingPlugin(
    #                         validation_stream=scenario.valid_stream,
    #                         patience=args.early_stopping,
    #                         check_each_n_iterations=args.eval_periodicity,
    #                         eval_mb_size=args.bs,
    #                         num_workers=args.num_workers)
    #         )
    #     else:
    #         metrics.append(EarlyStoppingPlugin(
    #                         validation_stream=scenario.valid_stream,
    #                         #validation_stream=scenario.test_stream,
    #                         patience=args.early_stopping,
    #                         eval_mb_size=args.bs,
    #                         num_workers=args.num_workers)
    #         )

    print("Plugins added...\n")
    return metrics


def get_optimizer(optim_name, 
                  model, 
                  lr, 
                  weight_decay=0.0, 
                  betas=(0.9,0.999), 
                  momentum=0.9,
                  ): #lr_classifier=None
    params = [{"params": model.parameters(), "lr": lr}]
    # if lr_classifier is not None:
    #     params = [{"params": model.feature_extractor.parameters(), "lr": lr},
    #               {"params": model.classifier.parameters(), "lr": lr_classifier}]
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, 
                                    weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, 
                                     weight_decay=weight_decay, betas=betas)
    elif optim_name == 'adamW':
        optimizer = torch.optim.AdamW(params, lr=lr, 
                                      weight_decay=weight_decay, betas=betas)
    else:
        print("No optimizer found for name", optim_name)
        raise ValueError()
    return optimizer

def get_strategy(args, model, eval_plugin, scenario, device, 
            plugins=None, data_transforms=None):
    plugins = [] if plugins is None else plugins

    # CRIT/OPTIM
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optim, 
                              model, 
                              args.lr, 
                              weight_decay=args.weight_decay, 
                              momentum=args.momentum) #lr_classifier=args.lr_classifier

    initial_epochs = args.epochs[0]

    # Use Iterations if defined
    if args.iterations_per_task is not None:
        args.epochs = [int(1e9)] # NOTE: something absurdly high to make sure we don't stop early
        initial_epochs = args.epochs[0]
        it_stopper = IterationsInsteadOfEpochs(max_iterations=args.iterations_per_task)
        plugins.append(it_stopper)
        print("\nUsing iterations instead of epochs, with", args.iterations_per_task, "iterations per task")


    # STRATEGY
    if args.strategy == 'finetune':
        strategy = Naive(model, optimizer, criterion,
                         train_epochs=initial_epochs, device=device,
                         train_mb_size=args.bs, evaluator=eval_plugin,
                         plugins=plugins
        )
        
    elif args.strategy == 'ER':
        print("\nUsing ER strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        ) 
        strategy.plugins.append(
                    ERPlugin(
                        n_total_memories=args.mem_size, 
                        device=device,
                        replay_batch_handling=args.replay_batch_handling,
                        task_incremental=args.task_incr,
                        domain_incremental=args.domain_incr,
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                    )
        )

    elif args.strategy == 'ER_vT':
        print("\nUsing ER strategy for timing")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        ) 
        strategy.plugins.append(
                    ERPluginVT(
                        n_total_memories=args.mem_size, 
                        sample_size=args.sample_size,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        task_incremental=(args.task_incr or args.domain_incr), 
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        use_replay_loss=True
                    )
        )
    
    elif args.strategy == 'ER_full':
        print("\nUsing ER_Full (incremental joint) strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        ) 
        strategy.plugins.append(
                    ERFullPlugin(
                        n_total_memories=args.mem_size, 
                        device=device,
                        replay_batch_handling=args.replay_batch_handling,
                        task_incremental=args.task_incr,
                        domain_incremental=args.domain_incr,
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                    )
        )

    elif args.strategy == "ER_full_GEM":
        print("\nUsing ER_Full_GEM strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        ) 
        strategy.plugins.append(
                    ERFullGEMPlugin(
                        n_total_memories=args.mem_size, 
                        lmbda=args.lmbda, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        memory_strength=args.gem_gamma
                    )
        )

    elif args.strategy == "ER_full_AGEM":
        print("\nUsing ER_Full_AGEM strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        ) 
        strategy.plugins.append(
                    ERFullAGEMPlugin(
                        n_total_memories=args.mem_size, 
                        lmbda=args.lmbda, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        num_experiences=scenario.n_experiences,
                    )
        )

    elif args.strategy == 'GEM':
        strategy = SupervisedTemplate(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=plugins
        )
        strategy.plugins.append(
            ERGEMPlugin(
                        n_total_memories=args.mem_size,
                        num_experiences=scenario.n_experiences,
                        total_num_classes=scenario.n_classes,
                        memory_strength=args.gem_gamma,
                        num_worker=args.num_workers,
                        task_incr=(args.task_incr or args.domain_incr),
                        use_replay_loss=False,
                        lmbda=args.lmbda,
                        do_decay_lmbda=args.do_decay_lmbda,
            )
        )

    elif args.strategy == 'GEMv2':
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
                    ERGEMPluginV2(
                        n_total_memories=args.mem_size,
                        num_experiences=scenario.n_experiences,
                        total_num_classes=scenario.n_classes,
                        memory_strength=args.gem_gamma,
                        num_worker=args.num_workers,
                        task_incr=(args.task_incr or args.domain_incr),
                        use_replay_loss=False,
                        lmbda=args.lmbda,
                        do_decay_lmbda=args.do_decay_lmbda,
                    )
        )

    elif args.strategy == 'ER_GEM':
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
                    ERGEMPlugin(
                        n_total_memories=args.mem_size,
                        num_experiences=scenario.n_experiences,
                        total_num_classes=scenario.n_classes,
                        #patterns_per_experience=(args.mem_size // scenario.n_experiences), 
                        memory_strength=args.gem_gamma,
                        num_worker=args.num_workers,
                        task_incr=(args.task_incr or args.domain_incr),
                        use_replay_loss=True,
                        lmbda=args.lmbda,
                        do_decay_lmbda=args.do_decay_lmbda,
                        small_const=args.gem_const
                    )
        )

    elif args.strategy == 'ER_GEMv2':
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
                    ERGEMPluginV2(
                        n_total_memories=args.mem_size,
                        num_experiences=scenario.n_experiences,
                        total_num_classes=scenario.n_classes,
                        memory_strength=args.gem_gamma,
                        num_worker=args.num_workers,
                        task_incr=(args.task_incr or args.domain_incr),
                        use_replay_loss=True,
                        lmbda=args.lmbda,
                        do_decay_lmbda=args.do_decay_lmbda,
                        small_const=args.gem_const
                    )
        )


    elif args.strategy == 'AGEM':
        print("\n Using AGEM strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins
        )
        strategy.plugins.append(
                    ERAGEMPlugin(
                        n_total_memories=args.mem_size, 
                        sample_size=args.sample_size,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        task_incremental=(args.task_incr or args.domain_incr), 
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        use_replay_loss=False
                    )
        )

    elif args.strategy == 'AGEMv2':
        print("\n Using AGEM strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins
        )
        strategy.plugins.append(
                    ERAGEMPluginV2(
                        n_total_memories=args.mem_size, 
                        sample_size=args.sample_size,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        task_incremental=(args.task_incr or args.domain_incr), 
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        use_replay_loss=False
                    )
        )

    elif args.strategy == 'ER_AGEM':
        print("\n Using ER_AGEM_custom strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
                    ERAGEMPlugin(
                        n_total_memories=args.mem_size, 
                        sample_size=args.sample_size,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        task_incremental=(args.task_incr or args.domain_incr), 
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        use_replay_loss=True
                    )
        )
    
    elif args.strategy == 'ER_AGEMv2':
        print("\n Using ER_AGEM_custom strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
                    ERAGEMPluginV2(
                        n_total_memories=args.mem_size, 
                        sample_size=args.sample_size,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        task_incremental=(args.task_incr or args.domain_incr), 
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        use_replay_loss=True
                    )
        )

    elif args.strategy == 'ER_AGEMvT':
        print("\n Using ER_AGEM strategy for timing")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
                    ERAGEMPluginVT(
                        n_total_memories=args.mem_size, 
                        sample_size=args.sample_size,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        task_incremental=(args.task_incr or args.domain_incr), 
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        use_replay_loss=True
                    )
        )

    elif args.strategy == 'BiC':
        print("\n Using BiC strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins
        )
        strategy.plugins.append(
            BiCPlugin(
                mem_size = args.mem_size,
                task_balanced_dataloader=args.task_balanced_dataloader,
                val_percentage=args.val_percentage,
                T = 2,
                stage_2_epochs = args.second_stage_eps,
                lr = args.second_stage_lr,
                lamb = args.lwf_alpha,
                num_workers = args.num_workers,
                verbose=False, 
                er_lmbda=args.lmbda,
                do_decay_er_lmbda=args.do_decay_lmbda,
                total_num_classes=scenario.n_classes,
                use_agem=False
            )
        )

    elif args.strategy == 'BiC_AGEM':
        print("\n Using BiC_AGEM strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
            BiCPlugin(
                mem_size = args.mem_size,
                task_balanced_dataloader=args.task_balanced_dataloader,
                val_percentage=args.val_percentage,
                T = 2,
                stage_2_epochs = args.second_stage_eps,
                lr = args.second_stage_lr,
                lamb = args.lwf_alpha,
                num_workers = args.num_workers,
                verbose=False, 
                er_lmbda=args.lmbda,
                do_decay_er_lmbda=args.do_decay_lmbda,
                total_num_classes=scenario.n_classes,
                use_agem=True
            )
        )

    elif args.strategy == 'DER':
        print("\n Using DER strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        # TODO:
        strategy.plugins.append(
            DERPlugin(mem_size=args.mem_size,
                      total_num_classes=scenario.n_classes,
                      batch_size_mem=strategy.train_mb_size,
                      alpha=0.5,  # Default
                      beta=0.0,
                      do_decay_beta=False, # Default
                      task_incremental=(args.task_incr or args.domain_incr),
                      num_experiences=scenario.n_experiences) 
        )
    
    elif args.strategy == 'DER++':
        print("\n Using DER strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
            DERPlugin(mem_size=args.mem_size,
                      total_num_classes=scenario.n_classes,
                      batch_size_mem=strategy.train_mb_size,
                      alpha=0.5,  # Default
                      beta=args.lmbda,
                      do_decay_beta=args.do_decay_lmbda, 
                      task_incremental=(args.task_incr or args.domain_incr),
                      num_experiences=scenario.n_experiences) 
        )

    elif args.strategy == 'DER_AGEM':
        print("\n Using DER strategy")
        strategy = SupervisedTemplate(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins,
        )
        strategy.plugins.append(
            DERAGEMPlugin(mem_size=args.mem_size,
                      total_num_classes=scenario.n_classes,
                      batch_size_mem=strategy.train_mb_size,
                      alpha=0.5,  # Default
                      beta=0,
                      do_decay_beta=False, # Default
                      use_agem=True,
                      task_incremental=(args.task_incr or args.domain_incr),
                      num_experiences=scenario.n_experiences)  
        )

    else:
        raise NotImplementedError(f"Non existing strategy arg: {args.strategy}")
    
    #################
    # Additional auxiliary plugins
    #################
    # LRScheduling
    if args.lr_scheduling:
        print("Using Multi-Step LRScheduler")
        scheduler = MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.1)
        strategy.plugins.append(LRSchedulerPlugin(
                scheduler,
                reset_lr=True,
                reset_scheduler=True,
                metric=None,
                step_granularity="epoch",
                first_epoch_only=False,
                first_exp_only=False,
            ))

    # Model storing to disk
    if args.store_models:
        strategy.plugins.append(StoreModelsPlugin(model_name=args.backbone, model_store_path=args.results_path))


    print(f"Running strategy:{strategy}")
    if hasattr(strategy, 'plugins'):
        print(f"with Plugins: {strategy.plugins}")
    return strategy


def overwrite_args_with_config(args, config_path):
    """
    Directly overwrite the input args with values defined in config yaml file.
    Only if args.config_path is defined.
    """
    if config_path is None:
        return
    assert os.path.isfile(config_path), f"Config file does not exist: {config_path}"

    import yaml
    with open(config_path, 'r') as stream:
        arg_configs = yaml.safe_load(stream)

    for arg_name, arg_val in arg_configs.items():  # Overwrite
        assert hasattr(args, arg_name), \
            f"'{arg_name}' defined in config is not specified in args, config: {config_path}"
        print(arg_name, arg_val)
        setattr(args, arg_name, arg_val)

    print(f"Overriden args with config: {config_path}")