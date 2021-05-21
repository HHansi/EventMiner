# Created by Hansi at 5/4/2021

# Monolingual binary classifier-based experiment
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from algo.coreference.sunlp_clustering import get_scores, get_clusters
from algo.language_modeling.language_modeling_model import LanguageModelingModel
from algo.sentence_transformers.classification_model import STClassificationSModel
from algo.coreference.coreference_model import CoreferenceModel
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments.common.evaluation import evaluate_clustering
from experiments.coreference_identifiation.data_formatter import sentence_pair_format
from experiments.coreference_identifiation.monolingual_config import DATA_DIRECTORY, config, MODEL_NAME, TEMP_DIRECTORY, \
    sp_config, SEED, LANGUAGE_FINETUNE, MODEL_TYPE, language_modeling_config, SUBMISSION_FILE, SUBMISSION_DIRECTORY
from experiments.common.data_util import read_data

delete_create_folder(TEMP_DIRECTORY)
create_folder_if_not_exist(SUBMISSION_DIRECTORY)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def save_clusters(predictions, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in predictions:
            f.write(json.dumps(doc) + "\n")


train_data = read_data((os.path.join(DATA_DIRECTORY, "en/train.json")))
dev_data = read_data((os.path.join(DATA_DIRECTORY, "en/dev.json")))
test_data = read_data((os.path.join(DATA_DIRECTORY, "en/test.json")))

if LANGUAGE_FINETUNE:
    print(f'Language modelling...')
    train_list = []
    for row in train_data:
        train_list.extend(row['sentences'])
    dev_list = []
    for row in dev_data:
        dev_list.extend(row['sentences'])
    test_list = []
    for row in test_data:
        test_list.extend(row['sentences'])

    complete_list = train_list + dev_list + test_list
    complete_list = list(set(complete_list))

    lm_train = complete_list[0: int(len(complete_list) * 0.8)]
    lm_test = complete_list[-int(len(complete_list) * 0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_config)
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"),
                      eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    MODEL_NAME = language_modeling_config["best_model_dir"]


if config['embedding_learning'] == 'pre-trained':
    model = CoreferenceModel(MODEL_NAME, args=config)
    eval = train_data

else:
    train, eval = train_test_split(train_data, test_size=0.1, random_state=SEED)
    train_df = sentence_pair_format(train)
    train_df = train_df.dropna()
    eval_df = sentence_pair_format(eval)
    eval_df = eval_df.dropna()

    sp_dev = sentence_pair_format(dev_data)
    sp_test = sentence_pair_format(test_data, has_labels=False)

    if config['embedding_learning'] == 'fine-tune':
        sp_model = STClassificationSModel(MODEL_NAME, args=sp_config)
    elif config['embedding_learning'] == 'from-scratch':
        sp_model = STClassificationSModel(MODEL_NAME, args=sp_config, embedding_learning='from-scratch')
    else:
        raise KeyError("Unknown embedding_learning type found!")

    sp_threshold = sp_model.train(train_df, eval_df)

    sp_model = STClassificationSModel(sp_config["best_model_dir"], args=sp_config, threshold=sp_threshold)
    sp_eval_preds = sp_model.predict(eval_df)
    print(f'Eval precision: {precision_score(list(eval_df["labels"].astype(int)), sp_eval_preds)}')
    print(f'Eval recall: {recall_score(list(eval_df["labels"].astype(int)), sp_eval_preds)}')
    print(f'Eval f1: {f1_score(list(eval_df["labels"].astype(int)), sp_eval_preds)}')

    sp_predictions = sp_model.predict(sp_dev)
    print(f'Precision: {precision_score(list(sp_dev["labels"].astype(int)), sp_predictions)}')
    print(f'Recall: {recall_score(list(sp_dev["labels"].astype(int)), sp_predictions)}')
    print(f'F1: {f1_score(list(sp_dev["labels"].astype(int)), sp_predictions)}')
    sp_dev["predictions"] = sp_predictions
    sp_dev.to_csv(os.path.join(SUBMISSION_DIRECTORY, "en-dev.csv"), index=False)

    model = CoreferenceModel(sp_config["best_model_dir"], args=config)

    # Approach 1 - SU-NLP method
    print(f'Approach 1 - SU-NLP method')
    scores = get_scores(sp_dev, reward=0.8, penalty=0.8)
    news_clusters = get_clusters(scores)
    predictions1 = [{"id": k, "pred_clusters": news_clusters[k]} for k in news_clusters.keys()]
    evaluate_clustering(dev_data, predictions1, config['eval_metric'])
    save_clusters(predictions1, file_path=os.path.join(SUBMISSION_DIRECTORY, "en-dev1.json"))

    test_sp_predictions = sp_model.predict(sp_test)
    sp_test["predictions"] = test_sp_predictions
    test_scores = get_scores(sp_test, reward=0.8, penalty=0.8)
    test_clusters = get_clusters(test_scores)
    test_predictions1 = [{"id": k, "pred_clusters": test_clusters[k]} for k in test_clusters.keys()]
    save_clusters(test_predictions1, file_path=os.path.join(TEMP_DIRECTORY, 'sunlp_' + SUBMISSION_FILE))


# Approach 2 - train HAC threshold using eval set
print(f'Approach 2 - train HAC threshold using eval set')
train_threshold, train_metric_value = model.train(eval)
print(f"trained threshold: {train_threshold} \t {config['eval_metric']}: {train_metric_value}")
predictions2 = model.predict(dev_data, train_threshold)
evaluate_clustering(dev_data, predictions2, config['eval_metric'])
save_clusters(predictions2, file_path=os.path.join(SUBMISSION_DIRECTORY, "en-dev2.json"))

# # Approach 3 - use average distance as threshold with HAC
# print(f'Approach 3 - use average distance as threshold with HAC')
# predictions3 = model.predict(dev_data)
# evaluate_clustering(dev_data, predictions3, config['eval_metric'])
# save_clusters(predictions3, file_path=os.path.join(SUBMISSION_DIRECTORY, "en-dev3.json"))

# Approach 4 - use dynamic tree cut
# print(f'Approach 4 - use dynamic tree cut')
# config['clustering'] = 'tree-cut'
# predictions4 = model.predict(dev_data, args=config)
# evaluate_clustering(dev_data, predictions4, config['eval_metric'])
# save_clusters(predictions4, file_path=os.path.join(SUBMISSION_DIRECTORY, "en-dev4.json"))

test_predictions = model.predict(test_data, train_threshold)
save_clusters(test_predictions, file_path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))

