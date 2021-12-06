from scipy.spatial import distance
from scipy.special import kl_div
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import numpy as np
import model_parameters

EPSILON = 1e-1

def get_acc_f1(test_trues, test_preds, num_classes=model_parameters.num_classes):
    total = 0
    correct = 0

    matches, gold, system = {}, {}, {}
    for i in range(num_classes):
        matches[i] = 0
        system[i] = 0
        gold[i] = 0

    for p, g in zip(test_preds,test_trues):
        total+=1
        if p == g:
            correct+=1
            matches[p] += 1

        gold[g] += 1
        system[p] += 1


    recall = {}
    precision = {}
    f1 = {}
    for i in range(num_classes):
        recall[i] = 1.0 * matches[i] / gold[i] if matches[i] != 0 else 0
        precision[i] = 1.0 * matches[i] / system[i] if matches[i] !=0 else 0
        f1[i] =  (2 * (precision[i] * recall[i])/(precision[i] + recall[i])) if (precision[i] + recall[i]) > 0 else 0

    support = np.array([gold[i] for i in range(num_classes)])

    average_recall = np.average([recall[i] for i in range(num_classes)], weights=support)
    average_recall = np.average([recall[i] for i in range(num_classes)], weights=support)
    average_precision = np.average([precision[i] for i in range(num_classes)], weights=support)
    average_f1 = np.average([f1[i] for i in range(num_classes)], weights=support)

    acc = correct/total

    return acc, average_precision, average_recall, average_f1



def get_ct_f1(test_trues, test_preds, test_distrs, num_classes=model_parameters.num_classes):

    tp, fp, fn, gold = {}, {}, {}, {}
    for i in range(num_classes):
        tp[i] = 0
        fp[i] = 0
        fn[i] = 0
        gold[i] = 0


    for p, g, distr in zip(test_preds, test_trues, test_distrs):
        # srs score of the gold not the predicted. Todo: talk it over with Massimo
        unit_vector = np.array([1 if i==g else 0 for i in range(num_classes)]).reshape(1,num_classes)
        distr = np.array(distr).reshape(1, num_classes)
        srs_s = cosine_similarity(unit_vector, distr)[0][0]
        if p == g:
            tp[p] += srs_s # correct hit
        else:
            fp[p] += (1-srs_s)  # miss
            fn[g] += srs_s   # correct rejection

        gold[g] += 1

    recall = {}
    precision = {}
    f1 = {}
    for i in range(num_classes):
        precision[i] = 1.0 * tp[i] / (tp[i] + fp[i]) if (fp[i] + tp[i]) != 0 else 0
        recall[i] = 1.0 * tp[i] / (tp[i] + fn[i]) if (fn[i] + tp[i]) !=0 else 0
        f1[i] =  (2 * (precision[i] * recall[i])/(precision[i] + recall[i])) if (precision[i] + recall[i]) > 0 else 0

    support = np.array([gold[i] for i in range(num_classes)])

    average_recall = np.average([recall[i] for i in range(num_classes)], weights=support)
    average_precision = np.average([precision[i] for i in range(num_classes)], weights=support)
    average_f1 = np.average([f1[i] for i in range(num_classes)], weights=support)

    return average_precision, average_recall, average_f1



def get_jsd_kl_div(soft_probs, predicted_probs):
    num_items = len(predicted_probs)
    all_jsd = [distance.jensenshannon(soft_probs[i], predicted_probs[i]) for i in range(num_items)]
    all_kl = [kl_div(soft_probs[i], predicted_probs[i]) for i in range(num_items)]
    return np.sum(all_jsd)/num_items, np.sum(all_kl)/num_items



def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce
