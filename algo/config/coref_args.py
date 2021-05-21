# Created by Hansi at 4/28/2021

from dataclasses import asdict, dataclass, field, fields


@dataclass
class CorefArgs:
    output_dir: str = 'temp/outputs/'
    threshold_step: float = 0.1
    threshold_min: float = 0
    threshold_max: float = 1
    affinity: str = 'cosine'
    linkage: str = 'average'  # 'single', 'complete'
    eval_metric: str = 'CoNLL-2012 average score'
    metric_minimize: bool = False
    embedding_learning: str = 'pre-trained'  # pre-trained, fine-tune, from-scratch
    clustering: str = 'hac'  # 'hac', 'tree-cut'

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))


@dataclass
class SPArgs:
    output_dir: str = "outputs/"
    best_model_dir: str = "outputs/best_model"
    num_train_epochs: int = 1
    warmup_steps: int = 10000
    learning_rate: float = 2e-5  # optimizer=AdamW
    weight_decay: float = 0.01
    evaluate_during_training_steps: int = 0
    max_grad_norm: float = 1.0
    show_progress_bar: bool = True
    train_batch_size: int = 8
    eval_batch_size: int = 8
    max_seq_length: int = 256
    do_lower_case: bool = False
    margin: float = 0.5  # only used with sp-classification
    loss_func: str = None  # default sp-classification- 'OnlineContrastiveLoss', other- 'MultipleNegativesRankingLoss'

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

