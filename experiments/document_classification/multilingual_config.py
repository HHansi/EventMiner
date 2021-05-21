import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

TEMP_DIRECTORY = "temp/data"
SUBMISSION_FILE = "submission.json"

MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-base"
LANGUAGES = ["en", "pr", "es"]


config = {
    'output_dir': 'temp/outputs/',
    "best_model_dir": "temp/outputs/best_model",
    'cache_dir': 'temp/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 512,
    'train_batch_size': 4,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 4,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 600,
    'save_steps': 600,
    "no_cache": False,
    'save_model_every_epoch': True,
    "save_recent_only": True,
    'n_fold': 3,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 200,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,

    'regression': False,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,

    "manual_seed": SEED,

    "encoding": None,
    "sliding_window": False
}
