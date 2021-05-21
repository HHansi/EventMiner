import itertools
import json
import subprocess


def convert_to_scorch_format(clusters):
    # Merge all documents' clusters in a single list

    all_clusters = []
    for idx, doc in enumerate(clusters):
        for cluster in doc:
            all_clusters.append([str(idx) + "_" + str(sent_id) for sent_id in cluster])

    all_events = [event for cluster in all_clusters for event in cluster]
    all_links = sum([list(itertools.combinations(cluster, 2)) for cluster in all_clusters], [])

    return all_links, all_events


def read_results(result_file):
    d = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            (key, val) = line.split(':')
            if '=' in val:
                for temp in val.split():
                    (sub_key, sub_val) = temp.split('=')
                    d[key + '-' + sub_key] = float(sub_val.strip())
            else:
                d[key] = float(val.strip())
    return d


def evaluate(gold_clusters, prediction_clusters):
    """
    Uses scorch -a python implementaion of CoNLL-2012 average score- for evaluation.
    > https://github.com/LoicGrobol/scorch | pip install scorch
    Takes gold file path (.json), predicted file path (.json) and prints out the results.

    :param gold_clusters: list of clusters
    :param prediction_clusters: list of clusters
    """

    gold_links, gold_events = convert_to_scorch_format(gold_clusters)
    sys_links, sys_events = convert_to_scorch_format(prediction_clusters)

    with open("gold.json", "w", encoding='utf-8') as f:
        json.dump({"type": "graph", "mentions": gold_events, "links": gold_links}, f)
    with open("sys.json", "w", encoding='utf-8') as f:
        json.dump({"type": "graph", "mentions": sys_events, "links": sys_links}, f)

    subprocess.run(["scorch", "gold.json", "sys.json", "results.txt"])
    results = read_results('results.txt')
    subprocess.run(["rm", "gold.json", "sys.json", "results.txt"])
    return results
