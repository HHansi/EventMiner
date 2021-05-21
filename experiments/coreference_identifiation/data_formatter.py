# Created by Hansi at 5/2/2021

import pandas as pd

from algo.util.file_util import create_folder_if_not_exist


def sentence_pair_format(data, file_path=None, has_labels=True):
    """
    :param data: list of dictionaries
    {'event_clusters':[[1,8]], 'sentence_no':[1,8], 'sentences':['sentence 1', 'sentence 2']}
    :return:
    """
    if has_labels:
        df = pd.DataFrame(columns=['id', 'text_a_id', 'text_b_id', 'text_a', 'text_b', 'labels'])
    else:
        df = pd.DataFrame(columns=['id', 'text_a_id', 'text_b_id', 'text_a', 'text_b'])

    n = 1
    for r in range(0, len(data)):
        row = data[r]
        for i in range(0, len(row['sentence_no'])):
            id_a = row['sentence_no'][i]
            if has_labels:
                cluster = None
                for temp in row['event_clusters']:
                    if id_a in temp:
                        cluster = temp
                        break
            for j in range(i + 1, len(row['sentence_no'])):
                id_b = row['sentence_no'][j]
                id = str(r) + '-' + str(id_a) + '-' + str(id_b)
                text_a = row['sentences'][i]
                text_b = row['sentences'][j]
                if has_labels:
                    if cluster is not None and id_b in cluster:
                        label = 1
                    else:
                        label = 0
                    df.loc[n] = [row['id'], id_a, id_b, text_a, text_b, label]
                else:
                    df.loc[n] = [row['id'], id_a, id_b, text_a, text_b]
                n += 1

    if file_path:
        create_folder_if_not_exist(file_path, is_file_path=True)
        df.to_csv(file_path, encoding='utf-8', index=False)
    return df
