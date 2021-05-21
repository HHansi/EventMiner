# Created by Hansi at 4/28/2021
import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')
SUBMISSION_DIRECTORY = os.path.join(BASE_PATH, 'submissions')
SUBMISSION_FILE = "submission.json"

TEMP_DIRECTORY = "temp"

MODEL_NAME = "bert-large-cased"
MODEL_TYPE = "bert"  # only used with language fine tuning
LANGUAGE_FINETUNE = False

config = {
    'output_dir': 'temp/outputs/',

    'threshold_step': 0.05,
    'threshold_min': 0.1,
    'threshold_max': 1,
    'affinity': 'cosine',
    'linkage': 'average',  # 'single', 'complete'

    'eval_metric': 'CoNLL-2012 average score',
    'metric_minimize': False,

    'embedding_learning': 'from-scratch',  # pre-trained, fine-tune, from-scratch
    # pre-trained, fine-tune - only supported for sentence transformers
}

sp_config = {
    'output_dir': 'temp/outputs/',
    "best_model_dir": "temp/outputs/best_model",

    'train_batch_size': 4,
    'eval_batch_size': 4,
    'num_train_epochs': 5,
    'warmup_steps': 0,
    'learning_rate': 1e-5,
    'weight_decay': 0,
    'evaluate_during_training_steps': 100,
    'max_grad_norm': 1.0,
    'max_seq_length': 136,  # 176,  # only used when embedding_learning=from-scratch
    'do_lower_case': False,
    'loss_func': None,

    'show_progress_bar': True
}

language_modeling_config = {
    'output_dir': 'temp/lm/outputs/',
    "best_model_dir": "temp/lm/outputs/best_model",
    'cache_dir': 'temp/lm/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 176,  # 136,  # 152
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 2,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 80,
    'save_steps': 80,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': True,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": True,
    'evaluate_during_training_steps': 80,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "config": {},
    "local_rank": -1,
    "encoding": None,

}
