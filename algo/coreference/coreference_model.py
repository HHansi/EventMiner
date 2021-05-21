# Created by Hansi at 4/28/2021
import os

import numpy as np
import pandas as pd

from numpy import arange
from scipy import spatial
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
from tqdm import tqdm

from algo.config.coref_args import CorefArgs
from algo.coreference.clustering import build_hac_clusters, get_ids_of_clusters, build_dynamic_cut_clusters
from algo.coreference.scorch_wrapper import evaluate
from algo.util.file_util import create_folder_if_not_exist


class CoreferenceModel:
    def __init__(
        self,
        model_name,
        args=None,
    ):
        """
        Initializes a CoreferenceModel
        :param model_name:
        :param args:
        """
        self.args = CorefArgs()

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, CorefArgs):
            self.args = args

        self.model = SentenceTransformer(model_name)

    def group_sentences(self, sentences, ids, threshold=None):
        embeddings = self.model.encode(sentences)
        if self.args.clustering == 'hac':
            cluster_model = build_hac_clusters(embeddings, threshold, self.args.affinity, self.args.linkage)
            groups = get_ids_of_clusters(cluster_model, ids)
        elif self.args.clustering == 'tree-cut':
            groups = build_dynamic_cut_clusters(embeddings, self.args.affinity, self.args.linkage, ids)
        else:
            raise KeyError(f'Unknown clustering type!')
        return groups

    def get_average_distance(self, sentences, affinity):
        embeddings = self.model.encode(sentences)
        distances = []
        for i in range(0, len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if affinity == 'cosine':
                    distances.append(spatial.distance.cosine(embeddings[i], embeddings[j]))
                else:
                    raise KeyError(f'Distance calculation is not supported for {affinity}')
        return np.mean(distances)

    def train(self, data):
        """
        :param data: list of dictionaries
            {'event_clusters':[[1,8]], 'sentence_no':[1,8], 'sentences':['sentence 1', 'sentence 2']}
        :return:
        """
        optimum_threshold = 0
        if self.args.metric_minimize:
            optimum_metric = 10000
        else:
            optimum_metric = 0

        df = pd.DataFrame(columns=['threshold', 'eval_metric'])

        i = 0
        for t in arange(self.args.threshold_min, self.args.threshold_max, self.args.threshold_step):
            print(f'training on threshold: {t}')
            i += 1
            temp_gold_clusters = []
            temp_pred_clusters = []
            for element in tqdm(data):
                groups = self.group_sentences(element['sentences'], element['sentence_no'], t)
                temp_gold_clusters.append(element['event_clusters'])
                temp_pred_clusters.append(list(groups.values()))

            eval_results = evaluate(temp_gold_clusters, temp_pred_clusters)
            temp_eval_metric = eval_results[self.args.eval_metric]
            print(f'eval results: {temp_eval_metric}')
            df.loc[i] = [t, temp_eval_metric]

            if self.args.metric_minimize and temp_eval_metric < optimum_metric:
                optimum_threshold = t
                optimum_metric = temp_eval_metric
            elif not self.args.metric_minimize and temp_eval_metric > optimum_metric:
                optimum_threshold = t
                optimum_metric = temp_eval_metric

        if self.args.output_dir:
            create_folder_if_not_exist(os.path.join(self.args.output_dir, 'coreference_model_train_eval.csv'), is_file_path=True)
            df.to_csv(os.path.join(self.args.output_dir, 'coreference_model_train_eval.csv'))
        return optimum_threshold, optimum_metric

    def predict(self, data, threshold=None, args=None):
        if args is not None and isinstance(args, dict):
            self.args.update_from_dict(args)

        predictions = []
        for element in data:
            temp_threshold = threshold
            if threshold is None and self.args.clustering != 'tree-cut':
                temp_threshold = self.get_average_distance(element['sentences'], self.args.affinity)

            groups = self.group_sentences(element['sentences'], element['sentence_no'], temp_threshold)

            # element['pred_clusters'] = groups
            predictions.append({"id": element["id"], "pred_clusters": list(groups.values())})
        return predictions


