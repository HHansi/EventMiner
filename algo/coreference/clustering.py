# Created by Hansi at 4/28/2021
import os

import joblib
from sklearn.cluster import AgglomerativeClustering

# from dynamicTreeCut import cutreeHybrid
# from scipy.spatial.distance import pdist
# import numpy as np
# from scipy.cluster.hierarchy import linkage


def build_hac_clusters(vectors, threshold, affinity, linkage, model_path=None):
    model = AgglomerativeClustering(distance_threshold=threshold, compute_full_tree=True, n_clusters=None,
                                      affinity=affinity, linkage=linkage)
    model.fit_predict(vectors)

    # save model if a model_name is given
    if model_path:
        model_path = model_path + ".joblib"
        # create folder if not exist
        folder_path = '/'.join(model_path.split('/')[0:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # save cluster model
        joblib.dump(model, model_path)

    return model


def get_ids_of_clusters(model, id_list):
    cluster_elements = dict()

    for i, word in enumerate(id_list):
        cluster_label = model.labels_[i]
        if cluster_label in cluster_elements:
            cluster_elements[cluster_label].append(word)
        else:
            cluster_elements[cluster_label] = [word]

    return cluster_elements


def build_dynamic_cut_clusters(vectors, affinity, linkage_type, id_list):
    # distances = pdist(vectors, affinity)
    # link = linkage(distances, linkage_type)
    # # link = linkage(distances, "average")
    # clusters = cutreeHybrid(link, distances, minClusterSize=2)
    #
    # if isinstance(clusters, dict):
    #     print(clusters["labels"])
    #     cluster_labels = clusters["labels"]
    # else:
    #     print(clusters)
    #     cluster_labels = clusters
    #
    # cluster_elements = dict()
    # for i, word in enumerate(id_list):
    #     cluster_label = cluster_labels[i]
    #     if cluster_label in cluster_elements:
    #         cluster_elements[cluster_label].append(word)
    #     else:
    #         cluster_elements[cluster_label] = [word]
    # return cluster_elements
    return None
