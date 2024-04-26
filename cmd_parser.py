import argparse
from distutils.util import strtobool

def get_arg_parser():
    parser = argparse.ArgumentParser()

    # Meta hyperparams
    parser.add_argument('--exp_name', default="Exp", type=str, help='Name for the experiment.')
    parser.add_argument('--benchmark_config', type=str, default=None,
                        help='Yaml file with config for the benchmark.')
    parser.add_argument('--strategy_config', type=str, default=None,
                        help='Yaml file with config for the strategy.')

    parser.add_argument('--exp_postfix', type=str, default='#now,#uid',
                        help='Extension of the experiment name. A static name enables continuing if checkpointing is define'
                            'Needed for fixes/gridsearches without needing to construct a whole different directory.'
                            'To use argument values: use # before the term, and for multiple separate with commas.'
                            'e.g. #cuda,#featsize,#now,#uid')
    parser.add_argument('--save_path', type=str, default='./results/', help='save eval results.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for the dataloaders.')
    parser.add_argument('--cuda', default=True, type=lambda x: bool(strtobool(x)), help='Enable cuda?')
    parser.add_argument('--disable_pbar', default=True, type=lambda x: bool(strtobool(x)), help='Disable progress bar')
    
    parser.add_argument('--n_seeds', default=5, type=int, help='Nb of seeds to run.')
    parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
    parser.add_argument('--deterministic', default=False, type=lambda x: bool(strtobool(x)),
                        help='Set deterministic option for CUDNN backend.')
    parser.add_argument('--wandb', default=False, type=lambda x: bool(strtobool(x)), 
                        help="Use wandb for exp tracking.")

    parser.add_argument('--requires_gpu', type=lambda x: bool(strtobool(x)), default=True, 
                        help='Marks the run to require a GPU. If no GPU is found, the run will not start.')
    parser.add_argument('--restrict_num_threads', default=None, type=int, 
                        help='Restrict number of threads for torch and numpy.')
    parser.add_argument('--restrict_num_interop_threads', default=None, type=int, 
                        help='Restrict number of threads for torch and numpy.')

    # Dataset
    parser.add_argument('--scenario', type=str, default='smnist',
                        choices=['smnist', 
                                 'rotmnist',
                                 'cifar100', 
                                 'domain_cifar100', 
                                 'miniimgnet',  
                                 ]
                        )
    parser.add_argument('--use_rand_class_ordering', action='store_true', default=False, 
                        help='Whether to use random task ordering.')
    parser.add_argument('--fixed_class_order', nargs='+', default=None, 
                        help='Fixed class order for the scenario.')
    parser.add_argument('--num_experiences', type=int, default=None, 
                        help='Number of experiences to use in the scenario.')
    parser.add_argument('--dset_rootpath', default='./data', type=str,
                        help='Root path of the downloaded dataset for e.g. Mini-Imagenet')  # Mini Imagenet
    parser.add_argument('--partial_num_tasks', type=int, default=None,
                        help='Up to which task to include, e.g. to consider only first 2 tasks of 5-task Split-MNIST')

    # Feature extractor
    parser.add_argument('--featsize', type=int, default=400,
                        help='The feature size output of the feature extractor.'
                            'The classifier uses this embedding as input.')
    parser.add_argument('--backbone', type=str, default='mlp', choices=['input', 
                                                                        'mlp', 
                                                                        'simple_cnn',
                                                                        'SlimResNet18'
                                                                        ], 
                        help="Feature extractor backbone.")
    parser.add_argument('--use_GAP', action='store_true', default=False,
                        help="Use Global Avg Pooling after feature extractor (for Resnet18).")
    parser.add_argument('--backbone_weights', type=str, default=None, help='Path to backbone weights.')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to custom pretrained weights.')
    parser.add_argument('--classifier_zero_init', action='store_true', default=False, help='Whether to zero init the classifier.')
    parser.add_argument('--overwrite_input_size', type=int, nargs='+', default=None, help='Overwrite data input_size the backbone and add respective transform to match data.')

    # Classifier
    parser.add_argument('--classifier', type=str, default='linear', choices=['linear', 
                                                                             'linear_dynamic',
                                                                             'norm_embed',
                                                                             'identity'
                                                                             ],  
                        help='linear classifier (prototype=weight vector for a class)'
                            'For feature-space classifiers, we output the embedding (identity) '
                            'or normalized embedding (norm_embed)')
    parser.add_argument('--lin_bias', default=True, type=lambda x: bool(strtobool(x)),
                        help="Use bias in Linear classifier")

    # Optimization
    parser.add_argument('--optim', type=str, choices=['sgd'], default='sgd')
    parser.add_argument('--bs', type=int, default=128, help='Minibatch size.')
    parser.add_argument('--epochs', type=int, nargs='+', default=[10], 
                        help='Number of epochs per experience. If len(epochs) != n_experiences, \
                            then the last epoch is used for the remaining experiences.')
    parser.add_argument('--iterations_per_task', type=int, default=None,
                        help='When this is defined, it overwrites the epochs per task.'
                            'This enables equal compute per task for imbalanced scenarios.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--lr_scheduling', action='store_true', default=False, help='Use learning rate scheduling.')
    parser.add_argument('--lr_milestones', type=str, default=None, help='Learning rate epoch decay milestones.')
    parser.add_argument('--lr_decay', type=float, default=None, help='Multiply factor on milestones.')

    parser.add_argument('--use_data_aug', nargs='+', default=[],
                        choices=['simclr', 'crop_flip', 'auto_cifar10', 'auto_imgnet', 'rand_crop', "corruption"],
                        help='Define one or more data augmentations. This is especially helpful with corruptions')
    parser.add_argument('--overwrite_mean', nargs='+', type=float, default=None, help='Overwrite mean_norm of dataset.')
    parser.add_argument('--overwrite_stddev', nargs='+', type=float, default=None, help='Overwrite std_norm of dataset.')

    # Continual Evaluation
    parser.add_argument('--eval_with_test_data', default=True, type=lambda x: bool(strtobool(x)),
                        help="Continual eval with the test or train stream, default True for test data of datasets.")
    parser.add_argument('--enable_continual_eval', default=True, type=lambda x: bool(strtobool(x)),
                        help='Enable evaluation each eval_periodicity iterations.')
    parser.add_argument('--eval_periodicity', type=int, default=1,
                        help='Periodicity in number of iterations for continual evaluation. (None for no continual eval)')
    parser.add_argument('--eval_task_subset_size', type=int, default=1000,
                        help='Max nb of samples per evaluation task. (-1 if not applicable)')
    
    parser.add_argument('--eval_max_iterations', type=int, default=-1, help='Max nb of iterations for continual eval.\
                        After this number of iters is reached, no more continual eval is performed. Default value \
                        of -1 means no limit.')
    parser.add_argument('--skip_initial_eval', action='store_true', default=False, help='Skip initial eval.')
    parser.add_argument('--only_initial_eval', action='store_true', default=False, help='Only perform initial eval.')
    parser.add_argument('--only_prepare_data', action='store_true', default=False, help='Only prepare data.')
    parser.add_argument('--terminate_after_exp', type=int, default=None, help='Terminate training after this experience.')

    # Expensive additional continual logging
    parser.add_argument('--track_extra_stats', action='store_true', default=False,
                        help="Track extra statistics, such as whether gradients are projected.")
    parser.add_argument('--track_class_stats', default=False, type=lambda x: bool(strtobool(x)),
                        help="To track per-class prototype statistics, if too many classes might be better to turn off.")
    parser.add_argument('--track_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the gradnorm of the evaluation tasks."
                            "This accumulates computational graphs from the entire task and is very expensive memory wise."
                            "Can be made more feasible with reducing 'eval_task_subset_size'.")
    parser.add_argument('--track_features', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the features before and after a single update. This is very expensive as "
                            "entire evaluation task dataset features are forwarded and stored twice in memory."
                            "Can be made more feasible with reducing 'eval_task_subset_size'.")
    parser.add_argument('--track_timing', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the timing of the forward and backward pass for each task.")

    # Strategy
    parser.add_argument('--strategy', type=str, default='finetune',
                        choices=['finetune',
                                 'ER', 'ER_full', 'ER_full_GEM', 'ER_full_AGEM',
                                 'GEM', 'GEMv2', 'ER_GEM',  'ER_GEMv2', 'ER_vT',
                                 'AGEM', 'AGEMv2', 'ER_AGEM', 'ER_AGEMv2', 'ER_AGEMvT', 
                                 'BiC', 'BiC_AGEM',
                                 'DER', 'DER_AGEM'
                                ],
                        help='Continual learning strategy.')
    parser.add_argument('--task_incr', action='store_true', default=False,
                        help="Give task ids during training to single out the head to the current task.")
    parser.add_argument('--domain_incr', action='store_true', default=False,
                        help="Rarely ever useful, but will make certain plugins behave like task-incremental\
                        which is desirable, e.g. for replay.")
    parser.add_argument('--eval_stored_weights', type=str, default=None,
                        help="When provided, will use the stored weights from the provided path to evaluate the model.")
    parser.add_argument('--rotations_list', type=float, nargs='+', default=None,
                        help='List of rotations to use for RotMNIST scenario.')
    
    # ER
    parser.add_argument('--Lw_new', type=float, default=0.5,
                        help='Weight for the CE loss on the new data, in range [0,1]')
    parser.add_argument('--record_stability_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                        help="Record the gradnorm of the memory samples in current batch?")
    parser.add_argument('--mem_size', default=1000, type=int, help='Total nb of samples in rehearsal memory.')
    parser.add_argument('--replay_batch_handling', choices=['separate', 'combined'], default='separate',
                        help='How to handle the replay batch. Separate means that the replay batch is handled \
                            separately from the current batch. Combined means that the replay batch is \
                            combined with the current batch.')

    # GENERAL
    parser.add_argument('--lmbda', type=float, default=1,
                        help='Lambda for weighting regularitzing loss terms.')

    # GEM
    parser.add_argument('--gem_gamma', default=0.5, type=float, help='Gem param to favor BWT')
    parser.add_argument('--gem_const', default=1e-3, type=float, help='GEM param for numerical stability')

    # AGEM
    parser.add_argument('--sample_size', default=0, type=int, 
                        help='Sample size to fine reference gradients for AGEM. Defaults to 0 which is deactivation.')
    
    # BiC
    parser.add_argument('--task_balanced_dataloader', action='store_true', default=False, 
                        help='Use task balanced dataloader for BiC method.')
    parser.add_argument('--val_percentage', type=float, default=0.1, 
                        help='Percentage of validation set for BiC method.')
    parser.add_argument('--second_stage_eps', type=int, default=10, 
                        help='Number of epochs for the second stage for BiC method.')
    parser.add_argument('--second_stage_lr', type=float, default=0.1, 
                        help='Learning rate for the second stage for BiC method.')

    # ER_AGEM
    parser.add_argument('--do_decay_lmbda', action='store_true', default=False, help='Decay lambda.')
    parser.add_argument('--lmbda_warmup', type=int, default=0, help='Number of warmup steps for lmbda in each experience.')

    # LWF
    parser.add_argument('--lwf_alpha', type=float, default=1, help='Distillation loss weight')
    parser.add_argument('--lwf_softmax_t', type=float, default=2, help='Softmax temperature (division).')

    # EWC
    parser.add_argument('--iw_strength', type=float, default=1, help="IW regularization strength.")

    # Store model every experience
    parser.add_argument('--store_models', action='store_true', default=False, help='Store model after each experience.')


    return parser