import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from algo.classification.classification_model import ClassificationModel
from experiments.common.evaluation import macro_f1, evaluate_classification
from experiments.common.data_util import read_data_df, clean_data, preprocess_data
from experiments.document_classification.multilingual_config import TEMP_DIRECTORY, SEED, LANGUAGES, DATA_DIRECTORY, \
    config, MODEL_TYPE, MODEL_NAME, SUBMISSION_FILE

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class TestInstance:
    def __init__(self, lang, df, sentences, preds):
        self.lang = lang
        self.df = df
        self.sentences = sentences
        self.preds = preds


train = pd.DataFrame(columns=['text', 'labels'])
# dev = pd.DataFrame(columns=['text', 'labels'])
dev_instances = dict()
test_instances = dict()

for lang in LANGUAGES:
    temp_train = read_data_df(os.path.join(DATA_DIRECTORY, lang, "train.json"))
    temp_train = clean_data(temp_train, text_column='text')
    temp_train = temp_train[['text', 'label']]
    temp_train = temp_train.rename(columns={'label': 'labels'})
    temp_train['text'] = temp_train['text'].apply(lambda x: preprocess_data(x))
    train = train.append(temp_train)

    temp_dev = read_data_df(os.path.join(DATA_DIRECTORY, lang, "dev.json"))
    temp_dev = clean_data(temp_dev, text_column='text')
    temp_dev = temp_dev[['text', 'label']]
    temp_dev = temp_dev.rename(columns={'label': 'labels'})
    temp_dev['text'] = temp_dev['text'].apply(lambda x: preprocess_data(x))
    dev_sentences = temp_dev['text'].tolist()
    dev_preds = np.zeros((len(dev_sentences), config["n_fold"]))
    dev_instances[lang] = TestInstance(lang, temp_dev, dev_sentences, dev_preds)
    # dev = dev.append(temp_dev)

    test = read_data_df(os.path.join(DATA_DIRECTORY, lang, "test.json"))
    test['text'] = test['text'].apply(lambda x: preprocess_data(x))
    test_sentences = test['text'].tolist()
    test_preds = np.zeros((len(test_sentences), config["n_fold"]))
    test_instances[lang] = TestInstance(lang, test, test_sentences, test_preds)

# Add Hindi data set for testing
lang = "hi"
test = read_data_df(os.path.join(DATA_DIRECTORY, lang, "test.json"))
test['text'] = test['text'].apply(lambda x: preprocess_data(x))
test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test_sentences), config["n_fold"]))
test_instances[lang] = TestInstance(lang, test, test_sentences, test_preds)

# shuffle data
train = train.sample(frac=1).reset_index(drop=True)
# dev = dev.sample(frac=1).reset_index(drop=True)

# dev_sentences = dev['text'].tolist()
# dev_preds = np.zeros((len(dev_sentences), config["n_fold"]))

for i in range(config["n_fold"]):
    if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
        shutil.rmtree(config['output_dir'])
    print("Started Fold {}".format(i))
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, f1=f1_score, recall=recall_score,
                      precision=precision_score)
    model = ClassificationModel(MODEL_TYPE, config["best_model_dir"], args=config,
                                use_cuda=torch.cuda.is_available())

    # predictions, raw_outputs = model.predict(dev_sentences)
    # dev_preds[:, i] = predictions

    for k in dev_instances.keys():
        predictions, raw_outputs = model.predict(dev_instances[k].sentences)
        dev_instances[k].preds[:, i] = predictions

    for k in test_instances.keys():
        test_predictions, test_raw_outputs = model.predict(test_instances[k].sentences)
        test_instances[k].preds[:, i] = test_predictions

    print("Completed Fold {}".format(i))

# select majority class of each instance (row)
# dev_predictions = []
# for row in dev_preds:
#     row = row.tolist()
#     dev_predictions.append(int(max(set(row), key=row.count)))
# dev["predictions"] = dev_predictions
# evaluate_classification(dev['labels'].tolist(), dev['predictions'].tolist())

for k in dev_instances.keys():
    dev_predictions = []
    for row in dev_instances[k].preds:
        row = row.tolist()
        dev_predictions.append(int(max(set(row), key=row.count)))  # select majority class of each instance (row)
    dev_instances[k].df["predictions"] = dev_predictions
    print(f'\n Evaluating {k}')
    evaluate_classification(dev_instances[k].df['labels'].tolist(), dev_instances[k].df['predictions'].tolist())

for k in test_instances.keys():
    test_predictions = []
    for row in test_instances[k].preds:
        row = row.tolist()
        test_predictions.append(int(max(set(row), key=row.count)))  # select majority class of each instance (row)
    test_instances[k].df["predictions"] = test_predictions

    with open(os.path.join(TEMP_DIRECTORY, k + '_' + SUBMISSION_FILE), 'w') as f:
        for index, row in test_instances[k].df.iterrows():
            item = {"id": row['id'], "prediction": row['predictions']}
            f.write("%s\n" % item)
