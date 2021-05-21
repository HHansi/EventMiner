# Created by Hansi at 4/9/2021
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from algo.util.file_util import create_folder_if_not_exist
from data_process.data_preprocessor import remove_links, remove_repeating_characters
from experiments.document_classification.monolingual_config import SEED, BASE_PATH
# from experiments.coreference_identifiation.en_config import SEED, BASE_PATH, DATA_DIRECTORY


def read_data(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data


def clean_data(data_df, text_column, min_seq_length=5, file_path=None):
    data_df = data_df.dropna()
    data_df['seq_length'] = data_df[text_column].apply(lambda x: len(x.split()))
    data_df = data_df[data_df['seq_length'] >= min_seq_length]
    if file_path:
        data_df.to_csv(file_path, index=False)
    return data_df


def read_data_df(path):
    data = read_data(path)
    return pd.DataFrame.from_records(data)


def save_data(data, path):
    create_folder_if_not_exist(path, is_file_path=True)
    with open(path, 'w') as f:
        for i in data:
            f.write("%s\n" % json.dumps(i))


def get_data_stat(df, text_column=None, label_column=None, plot_path=None):
    if label_column:
        # get label distribution
        temp_df = df.groupby(label_column).count()
        print(temp_df)

    if text_column:
        # get sequence length details
        if 'seq_length' not in df:
            df['seq_length'] = df[text_column].apply(lambda x: len(x.split()))
        print(df['seq_length'].describe())

        # draw histogram
        binwidth = 50
        df['seq_length'].plot.hist(grid=True,
                                   bins=range(min(df['seq_length']), max(df['seq_length']) + binwidth, binwidth),
                                   rwidth=0.5, color='#607c8e')
        plt.xlabel('Sequence Length')
        plt.ylabel('Counts')
        plt.grid(axis='y', alpha=0.75)
        # plt.xticks(np.arange(0, 3000, 250))
        if plot_path is not None:
            plt.savefig(plot_path)
        plt.show()


def split_data(path, train_path, dev_path, text_column, min_seq_length=None):
    data = read_data(path)

    if min_seq_length is not None:
        filtered_data = []
        for instance in data:
            if len(instance[text_column].split()) >= min_seq_length:
                filtered_data.append(instance)
        data = filtered_data

    train, dev = train_test_split(data, test_size=0.2, random_state=SEED)
    save_data(train, train_path)
    save_data(dev, dev_path)


def preprocess_data(text):
    text = remove_links(text, substitute='')
    text = remove_repeating_characters(text)
    # remove white spaces at the beginning and end of the text
    text = text.strip()
    # remove extra whitespace, newline, tab
    text = ' '.join(text.split())
    return text


if __name__ == '__main__':
    # data_file = os.path.join(BASE_PATH, 'data/en-train.json')
    data_file = os.path.join(BASE_PATH, 'data/es/dev.json')
    data_df = read_data_df(data_file)
    # data_df = clean_data(data_df, text_column='sentence')
    plot_path = os.path.join(BASE_PATH, 'results/es-dev.png')
    # plot_path = None
    get_data_stat(data_df, text_column='text', label_column=None, plot_path=plot_path)

    # train_path = os.path.join(BASE_PATH, 'data/en/train.json')
    # dev_path = os.path.join(BASE_PATH, 'data/en/dev.json')
    # split_data(data_file, train_path, dev_path, text_column='sentence', min_seq_length=5)
