import os

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')
SUBMISSION_DIRECTORY = os.path.join(BASE_PATH, 'submissions')
SUBMISSION_FILE = "submission.json"

TEMP_DIRECTORY = "temp"

MODEL_NAME = "quora-distilbert-multilingual"
TRAIN_LANGUAGE = "en"
TEST_LANGUAGES = ["pr", "es"]

config = {
   'output_dir': 'temp/outputs/',

   'threshold_step': 0.05,
   'threshold_min': 0.1,
   'threshold_max': 1,
   'affinity': 'cosine',
   'linkage': 'average',  # 'single', 'complete'

   'eval_metric': 'CoNLL-2012 average score',
   'metric_minimize': False,

   'embedding_learning': 'pre-trained',  # pre-trained, fine-tune, from-scratch
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
    'max_seq_length': 176,  # 136,  # only used when embedding_learning=from-scratch
    'do_lower_case': False,
    'loss_func': None,

    'show_progress_bar': True
}