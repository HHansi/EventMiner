import shutil

import numpy as np
import random
import os

import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from algo.classification.classification_model import ClassificationModel
from experiments.common.class_balancer import binary_class_balance
from experiments.common.evaluation import macro_f1, evaluate_classification
from experiments.common.data_util import read_data_df, clean_data, preprocess_data
from experiments.document_classification.monolingual_config import TEMP_DIRECTORY, DATA_DIRECTORY, config, MODEL_TYPE, \
    MODEL_NAME, SEED, SUBMISSION_FILE, LANGUAGE, BINARY_CLASS_BALANCE, CLASS, PROPORTION

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train = read_data_df(os.path.join(DATA_DIRECTORY, LANGUAGE, "train.json"))
train = clean_data(train, text_column='text')
train = train[['text', 'label']]
train = train.rename(columns={'label': 'labels'})
train['text'] = train['text'].apply(lambda x: preprocess_data(x))
if BINARY_CLASS_BALANCE:
    train = binary_class_balance(train, label_column='labels', label=CLASS, proportion=PROPORTION)

dev = read_data_df(os.path.join(DATA_DIRECTORY, LANGUAGE, "dev.json"))
dev = clean_data(dev, text_column='text')
dev = dev[['text', 'label']]
dev = dev.rename(columns={'label': 'labels'})
dev['text'] = dev['text'].apply(lambda x: preprocess_data(x))

test = read_data_df(os.path.join(DATA_DIRECTORY, LANGUAGE, "test.json"))
test['text'] = test['text'].apply(lambda x: preprocess_data(x))

dev_sentences = dev['text'].tolist()
dev_preds = np.zeros((len(dev_sentences), config["n_fold"]))

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test_sentences), config["n_fold"]))

for i in range(config["n_fold"]):
    if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
        shutil.rmtree(config['output_dir'])
    print("Started Fold {}".format(i))
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, f1=f1_score, recall=recall_score, precision=precision_score)
    model = ClassificationModel(MODEL_TYPE, config["best_model_dir"], args=config,
                                use_cuda=torch.cuda.is_available())

    predictions, raw_outputs = model.predict(dev_sentences)
    dev_preds[:, i] = predictions

    test_predictions, test_raw_outputs = model.predict(test_sentences)
    test_preds[:, i] = test_predictions

    print("Completed Fold {}".format(i))

# select majority class of each instance (row)
dev_predictions = []
for row in dev_preds:
    row = row.tolist()
    dev_predictions.append(int(max(set(row), key=row.count)))
dev["predictions"] = dev_predictions
evaluate_classification(dev['labels'].tolist(), dev['predictions'].tolist())

# select majority class of each instance (row)
test_predictions = []
for row in test_preds:
    row = row.tolist()
    test_predictions.append(int(max(set(row), key=row.count)))
test["predictions"] = test_predictions

with open(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'w') as f:
    for index, row in test.iterrows():
        item = {"id": row['id'], "prediction": row['predictions']}
        f.write("%s\n" % item)



