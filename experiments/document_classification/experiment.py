import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from algo.classification.classification_model import ClassificationModel
from algo.util.file_util import create_folder_if_not_exist
from experiments.common.class_balancer import binary_class_balance
from experiments.common.data_util import read_data_df, clean_data, preprocess_data
from experiments.common.evaluation import macro_f1
from experiments.document_classification.config import TEMP_DIRECTORY, DATA_DIRECTORY, config, MODEL_TYPE, \
    MODEL_NAME, SEED, SUBMISSION_FILE, BINARY_CLASS_BALANCE, CLASS, PROPORTION, TEST_LANGUAGES, TRAIN_LANGUAGES, \
    PREDICTION_DIRECTORY

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)


class TestInstanceClassifier:
    def __init__(self, lang, df, texts, preds):
        self.lang = lang
        self.df = df
        self.texts = texts
        self.preds = preds


def prepare_multilingual_data(languages, input_folder, seed):
    print(f"Preparing multilingual training data...")
    train_df = pd.DataFrame(columns=['text', 'labels'])
    eval_df = pd.DataFrame(columns=['text', 'labels'])

    for lang in languages:
        temp_train = read_data_df(os.path.join(input_folder, lang, "train.json"))
        temp_train = clean_data(temp_train, text_column='text')
        temp_train = temp_train[['text', 'label']]
        temp_train = temp_train.rename(columns={'label': 'labels'})
        temp_train['text'] = temp_train['text'].apply(lambda x: preprocess_data(x))
        if BINARY_CLASS_BALANCE:
            temp_train = binary_class_balance(temp_train, label_column='labels', label=CLASS, proportion=PROPORTION)

        temp_train_df, temp_eval_df = train_test_split(temp_train, test_size=0.1, random_state=SEED * i)
        print(f"{lang} split: {len(temp_train_df)}-{len(temp_eval_df)}")
        train_df = train_df.append(temp_train_df)
        eval_df = eval_df.append(temp_eval_df)

    # shuffle train and eval sets
    train_df = shuffle(train_df, random_state=seed)
    eval_df = shuffle(eval_df, random_state=seed)
    print(f"final split sizes: {len(train_df)}-{len(eval_df)}")

    return train_df, eval_df


test_instances = dict()
if TEST_LANGUAGES is not None:
    for lang in TEST_LANGUAGES:
        test = read_data_df(os.path.join(DATA_DIRECTORY, lang, "test.json"))
        test['text'] = test['text'].apply(lambda x: preprocess_data(x))
        test_texts = test['text'].tolist()
        test_preds = np.zeros((len(test_texts), config["n_fold"]))
        test_instances[lang] = TestInstanceClassifier(lang, test, test_texts, test_preds)

if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
    shutil.rmtree(config['output_dir'])
base_output_dir = config['output_dir']

for i in range(config["n_fold"]):
    print("Started Fold {}".format(i))
    config['output_dir'] = f"{base_output_dir}_{i}"
    config["best_model_dir"] = os.path.join(config['output_dir'], "model")

    seed = int(SEED * (i + 1))
    config["manual_seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config,
                                use_cuda=torch.cuda.is_available(), cuda_device=3)
    # train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    train_df, eval_df = prepare_multilingual_data(TRAIN_LANGUAGES, DATA_DIRECTORY, seed)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, f1=f1_score, recall=recall_score,
                      precision=precision_score)
    model = ClassificationModel(MODEL_TYPE, config["best_model_dir"], args=config,
                                use_cuda=torch.cuda.is_available(), cuda_device=3)

    print(f"Making test predictions for fold {i}...")
    for lang in test_instances.keys():
        predictions, raw_predictions = model.predict(test_instances[lang].texts)
        test_instances[lang].preds[:, i] = predictions

    print("Completed Fold {}".format(i))

# final test predictions
create_folder_if_not_exist(PREDICTION_DIRECTORY)
for lang in test_instances.keys():
    print(f"Calculating majority class for {lang}...")
    test_predictions = []
    for row in test_instances[lang].preds:
        row = row.tolist()
        test_predictions.append(int(max(set(row), key=row.count)))  # select majority class of each instance (row)
    test_instances[lang].df["predictions"] = test_predictions

    print(f"Saving test predictions for {lang}...")
    submission_file_name = os.path.basename(SUBMISSION_FILE)
    submission_file_name_splits = os.path.splitext(submission_file_name)
    submission_file = os.path.join(os.path.dirname(SUBMISSION_FILE),
                                   f"{submission_file_name_splits[0]}_{lang}{submission_file_name_splits[1]}")
    with open(submission_file, 'w') as f:
        for index, row in test_instances[lang].df.iterrows():
            item = {"id": row['id'], "prediction": row['predictions']}
            f.write("%s\n" % item)
