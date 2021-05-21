# Created by Hansi at 5/4/2021
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from algo.coreference.coreference_model import CoreferenceModel
from algo.coreference.sunlp_clustering import get_scores, get_clusters
from algo.sentence_transformers.classification_model import STClassificationSModel
from algo.util.file_util import delete_create_folder, create_folder_if_not_exist
from experiments.common.evaluation import evaluate_clustering
from experiments.coreference_identifiation.data_formatter import sentence_pair_format
from experiments.coreference_identifiation.multilingual_config import TRAIN_LANGUAGE, TEST_LANGUAGES, sp_config, \
    TEMP_DIRECTORY, SUBMISSION_DIRECTORY, SEED, DATA_DIRECTORY, MODEL_NAME, config, SUBMISSION_FILE

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


train_data = read_data((os.path.join(DATA_DIRECTORY, TRAIN_LANGUAGE, "train.json")))
dev_instances = dict()
test_instances = dict()
dev_instances[TRAIN_LANGUAGE] = read_data((os.path.join(DATA_DIRECTORY, TRAIN_LANGUAGE, "dev.json")))
test_instances[TRAIN_LANGUAGE] = read_data((os.path.join(DATA_DIRECTORY, TRAIN_LANGUAGE, "test.json")))

for lang in TEST_LANGUAGES:
    dev_instances[lang] = read_data((os.path.join(DATA_DIRECTORY, lang, "train.json")))
for lang in TEST_LANGUAGES:
    test_instances[lang] = read_data((os.path.join(DATA_DIRECTORY, lang, "test.json")))


if config['embedding_learning'] == 'pre-trained':
    model = CoreferenceModel(MODEL_NAME, args=config)
    eval = train_data

else:
    train, eval = train_test_split(train_data, test_size=0.1, random_state=SEED)
    train_df = sentence_pair_format(train)
    train_df = train_df.dropna()
    eval_df = sentence_pair_format(eval)
    eval_df = eval_df.dropna()

    sp_dev = sentence_pair_format(dev_instances[TRAIN_LANGUAGE])
    sp_test = sentence_pair_format(test_instances[TRAIN_LANGUAGE], has_labels=False)

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
    evaluate_clustering(dev_instances[TRAIN_LANGUAGE], predictions1, config['eval_metric'])
    save_clusters(predictions1, file_path=os.path.join(SUBMISSION_DIRECTORY, "en-dev1.json"))

    test_sp_predictions = sp_model.predict(sp_test)
    sp_test["predictions"] = test_sp_predictions
    test_scores = get_scores(sp_test, reward=0.8, penalty=0.8)
    test_clusters = get_clusters(test_scores)
    test_predictions1 = [{"id": k, "pred_clusters": test_clusters[k]} for k in test_clusters.keys()]
    save_clusters(test_predictions1, file_path=os.path.join(TEMP_DIRECTORY, 'sunlp_' + TRAIN_LANGUAGE + "_" + SUBMISSION_FILE))

# Approach 2 - train HAC threshold using eval set (for training language)
print(f'Approach 2 - train HAC threshold using eval set')
train_threshold, train_metric_value = model.train(eval)
print(f"trained threshold for {TRAIN_LANGUAGE}: {train_threshold} \t {config['eval_metric']}: {train_metric_value}")
predictions2 = model.predict(dev_instances[TRAIN_LANGUAGE], train_threshold)
evaluate_clustering(dev_instances[TRAIN_LANGUAGE], predictions2, config['eval_metric'])
save_clusters(predictions2, file_path=os.path.join(SUBMISSION_DIRECTORY, "en-dev2.json"))

# test predictions for trained language
test_predictions = model.predict(test_instances[TRAIN_LANGUAGE], train_threshold)
save_clusters(test_predictions, file_path=os.path.join(TEMP_DIRECTORY, TRAIN_LANGUAGE + "_" + SUBMISSION_FILE))

# test predictions for other languages
for lang in TEST_LANGUAGES:
    train_threshold, train_metric_value = model.train(dev_instances[lang])
    print(f"trained threshold for {lang}: {train_threshold} \t {config['eval_metric']}: {train_metric_value}")

    test_predictions = model.predict(test_instances[lang], train_threshold)
    save_clusters(test_predictions, file_path=os.path.join(TEMP_DIRECTORY, lang + '_' + SUBMISSION_FILE))

