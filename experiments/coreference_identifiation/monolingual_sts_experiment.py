# Created by Hansi at 4/28/2021

# Monolingual semantic textual similarity-based experiment
import json
import os
import random

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

from algo.sentence_transformers.regression_model import STRegressionModel
from algo.coreference.coreference_model import CoreferenceModel
from algo.coreference.scorch_wrapper import evaluate
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments.coreference_identifiation.data_formatter import sentence_pair_format
from experiments.coreference_identifiation.monolingual_config import DATA_DIRECTORY, config, MODEL_NAME, TEMP_DIRECTORY, \
    sp_config, SEED, SUBMISSION_DIRECTORY
from experiments.common.data_util import read_data

delete_create_folder(TEMP_DIRECTORY)
create_folder_if_not_exist(SUBMISSION_DIRECTORY)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

language = "en"

train_data = read_data((os.path.join(DATA_DIRECTORY, language, "train.json")))
dev_data = read_data((os.path.join(DATA_DIRECTORY, language, "dev.json")))
# test_data = read_data((os.path.join(DATA_DIRECTORY, "en-test.json")))

if config['embedding_learning'] == 'pre-trained':
    model = CoreferenceModel(MODEL_NAME, args=config)
    train_threshold, train_metric_value = model.train(train_data)
    dev_preds = model.predict(dev_data, train_threshold)

else:
    train, eval = train_test_split(train_data, test_size=0.1, random_state=SEED)
    train_df = sentence_pair_format(train)
    train_df = train_df.dropna()
    eval_df = sentence_pair_format(eval)
    eval_df = eval_df.dropna()

    sp_dev = sentence_pair_format(dev_data)

    if config['embedding_learning'] == 'fine-tune':
        sp_model = STRegressionModel(MODEL_NAME, args=sp_config)
    elif config['embedding_learning'] == 'from-scratch':
        sp_model = STRegressionModel(MODEL_NAME, args=sp_config, embedding_learning='from-scratch')
    else:
        raise KeyError("Unknown embedding_learning type found!")

    sp_model.train(train_df, eval_df)

    sp_model = STRegressionModel(sp_config["best_model_dir"], args=sp_config)
    sp_eval_preds = sp_model.predict(eval_df)
    eval_pearson_cosine, _ = pearsonr(list(eval_df["labels"].astype(float)), sp_eval_preds)
    eval_spearman_cosine, _ = spearmanr(list(eval_df["labels"].astype(float)), sp_eval_preds)
    print(f'Pearson Correlation-Eval: {eval_pearson_cosine}')
    print(f'Spearman Correlation-Eval: {eval_spearman_cosine}')

    sp_predictions = sp_model.predict(sp_dev)
    eval_pearson_cosine, _ = pearsonr(list(sp_dev["labels"].astype(float)), sp_predictions)
    eval_spearman_cosine, _ = spearmanr(list(sp_dev["labels"].astype(float)), sp_predictions)
    print(f'Pearson Correlation: {eval_pearson_cosine}')
    print(f'Spearman Correlation: {eval_spearman_cosine}')

    model = CoreferenceModel(sp_config["best_model_dir"], args=config)
    train_threshold, train_metric_value = model.train(train_data)

print(f"threshold: {train_threshold}")
print(f"{config['eval_metric']}: {train_metric_value}")

print("Evaluating dev set")
predictions = model.predict(dev_data, train_threshold)
gold_clusters = [temp['event_clusters'] for temp in dev_data]
pred_clusters = [temp['pred_clusters'] for temp in predictions]
eval_results = evaluate(gold_clusters, pred_clusters)
eval_metric = eval_results[config['eval_metric']]
print(f'dev results: {eval_metric}')

with open(os.path.join(SUBMISSION_DIRECTORY, "en-dev.json"), "w", encoding="utf-8") as f:
    for doc in predictions:
        f.write(json.dumps(doc) + "\n")

# predictions = model.predict(test_data)
# with open(os.path.join(SUBMISSION_DIRECTORY, "sample_submission.json"), "w", encoding="utf-8") as f:
#     for doc in predictions:
#         f.write(json.dumps(doc) + "\n")
