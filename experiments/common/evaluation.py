# Created by Hansi at 7/3/2020
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from algo.coreference.scorch_wrapper import evaluate


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def confusion_matrix_values(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    return confusion_matrix(y_pred, y_true).ravel()


def evaluate_classification(actuals, predictions):
    print("Precision: ", precision_score(actuals, predictions))
    print("Recall: ", recall_score(actuals, predictions))
    print("F1: ", f1_score(actuals, predictions))
    print("Macro Precision: ", precision_score(actuals, predictions, average='macro'))
    print("Macro Recall: ", recall_score(actuals, predictions, average='macro'))
    print("Macro F1: ", macro_f1(actuals, predictions))

    tn, fp, fn, tp = confusion_matrix_values(actuals, predictions)
    print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))


def evaluate_clustering(actuals, predictions, metric):
    """
    :param actuals: JSON objects
    :param predictions:
    :return:
    """
    gold_clusters = [temp['event_clusters'] for temp in actuals]
    pred_clusters = [temp['pred_clusters'] for temp in predictions]
    eval_results = evaluate(gold_clusters, pred_clusters)
    eval_metric = eval_results[metric]
    print(f'dev results: {eval_metric}')