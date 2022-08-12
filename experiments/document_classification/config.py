import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

TEMP_DIRECTORY = os.path.join(BASE_PATH, 'temp')
OUTPUT_DIRECTORY = os.path.join(TEMP_DIRECTORY, 'output')

PREDICTION_DIRECTORY = os.path.join(TEMP_DIRECTORY, 'predictions')
SUBMISSION_FILE = os.path.join(PREDICTION_DIRECTORY, 'submission.json')

# MODEL_TYPE = "xlmroberta"  # "xlmroberta"  # "auto"  # "bigbird"
# # model name or directory
# MODEL_NAME = "xlm-roberta-large"  # "dccuchile/bert-base-spanish-wwm-cased"  # "neuralmind/bert-large-portuguese-cased"

MODEL_TYPE = "auto"
# model name or directory
MODEL_NAME = "/experiments/tranasinghe/EventMiner/trained_models/longformer-900/0/model/"


# list of one or more languages for training and testing
TRAIN_LANGUAGES = ["es"]
TEST_LANGUAGES = ["es"]

BINARY_CLASS_BALANCE = False
CLASS = 0  # will be used if BINARY_CLASS_BALANCE=True
PROPORTION = 0.75  # proportion of the given CLASS expected in final dataset (will be used if BINARY_CLASS_BALANCE=True)

CUDA_DEVICE = 1

config = {
    'output_dir': OUTPUT_DIRECTORY,
    'best_model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),
    'cache_dir': 'temp/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 256,
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

    'logging_steps': 40,  #200,
    'save_steps': 40,  #200,
    "no_cache": False,
    'save_model_every_epoch': True,
    "save_recent_only": True,
    'n_fold': 3,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 40,  #200,
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


