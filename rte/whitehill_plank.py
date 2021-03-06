# -*- coding: utf-8 -*-
"""softmax_whitehill_plank_rte.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Ko_6ukUHQMHIkLX_QldOunYNqDlE60m
"""

import csv
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.special import kl_div
from scipy.stats import entropy
import torch
from tabulate import tabulate

torch.cuda.is_available()

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
sentence_encoder = SentenceTransformer('bert-base-nli-mean-tokens')

from google.colab import drive

drive.mount('/content/drive')

DATAPATH = 'drive/My Drive/Data/rte'

all_data = []

with open(DATAPATH+"/rte.standardized.tsv",encoding='utf8' ) as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
        if row != []:
            all_data.append(row)

all_responses = {}
gold_responses = {}

for annotation in all_data[1:]:
    task_id = annotation[2]
    annotator = annotation[1]
    response = annotation[-2]
    gold = annotation[-1]
    if task_id not in gold_responses:
        gold_responses[task_id] = int(gold)
        all_responses[task_id] = {annotator:int(response)}
    else:
        all_responses[task_id][annotator] = int(response)

whitehill = {}

with open(DATAPATH+"/rte-wh-estimates.csv",encoding='utf8' ) as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
        if row != []:
            idd, _, _, _, diff = row[0].split(',')
            whitehill[idd] = diff

len(whitehill)

dd = []
with open(DATAPATH+"/rte1.tsv",encoding='utf8' ) as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
        if row != []:
            dd.append(row)
            
data = {}
for id_, val, task, text, hyp in dd:
    data[id_] = [text, hyp]

from scipy.special import softmax

# creating the train and development data.
train_text = []
train_hypothesis = []
train_answers = []
train_gold = []
train_maj = []
train_distr = []
train_soft = []
train_diff = []

for item, annotation in all_responses.items():
    train_diff.append(float(whitehill[item]))

    crowd_labels = list(annotation.values())
    num_annotations = len(crowd_labels)
    train_answers.append(crowd_labels)
    train_text.append(data[item][0])
    train_hypothesis.append(data[item][1])
    train_gold.append(gold_responses[item])
    maj = max(crowd_labels,key=crowd_labels.count)
    train_maj.append(maj)
    distr = [crowd_labels.count(0), crowd_labels.count(1)]
    train_distr.append(distr)
    train_soft.append(softmax(distr))

train_diff[:4]

filename = 'drive/My Drive/Data/rte/ds_posterior.npy'
train_ds_posterior = np.load(filename).tolist()
train_ds = np.argmax(train_ds_posterior, 1).tolist()

embedded_text = sentence_encoder.encode(train_text)
embedded_hypothesis = sentence_encoder.encode(train_hypothesis)

embedded_text = sentence_encoder.encode(train_text)
embedded_hypothesis = sentence_encoder.encode(train_hypothesis)

norm = [0.5, 0.5]
item_entropys = [entropy(scores)/entropy(norm) for scores in train_soft]
item_entropys[0:5]

train_gold.count(0), train_gold.count(1), train_maj.count(0), train_maj.count(1), train_ds.count(0), train_ds.count(1)

def get_ct_f1(test_trues, test_preds, test_distrs, num_classes=2):
    total = 0
    correct = 0
    tp = {0:0, 1:0}
    fp = {0:0, 1:0}
    fn = {0:0, 1:0}
    gold = {0:0, 1:0}
    
    for p, g, distr in zip(test_preds, test_trues, test_distrs):
        if p > 0.5:
            p = 1
        else:
            p = 0
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

def get_acc_f1(test_trues, test_preds):
    total = 0
    correct = 0
    matches = {0:0, 1:0}
    gold = {0:0, 1:0}
    system = {0:0, 1:0}

    for p, g in zip(test_preds,test_trues):
        total+=1
        if p > 0.5:
            p = 1
        else:
            p = 0
        if p == g:
            correct+=1
            matches[p] += 1

        gold[g] += 1
        system[p] += 1
    
    
    recall = {}
    precision = {}
    f1 = {}
    for i in range(2):
        recall[i] = 1.0 * matches[i] / gold[i] if matches[i] != 0 else 0
        precision[i] = 1.0 * matches[i] / system[i] if matches[i] !=0 else 0
        f1[i] =  (2 * (precision[i] * recall[i])/(precision[i] + recall[i])) if (precision[i] + recall[i]) > 0 else 0

    support = np.array([gold[0], gold[1]])

    average_recall = np.average([recall[i] for i in range(2)], weights=support)
    average_precision = np.average([precision[i] for i in range(2)], weights=support)
    average_f1 = np.average([f1[i] for i in range(2)], weights=support)
    
    acc = correct/total

    return acc, average_precision, average_recall, average_f1

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

embedded_text = embedded_text.tolist()
embedded_hypothesis = embedded_hypothesis.tolist()

class RTE_model(torch.nn.Module):
    def __init__(self, mtype='stl', smythe='kl'):
        super().__init__()

        # concat_size = embedded_hypothesis[0].shape[0]*4
        # final_size = embedded_hypothesis[0].shape[0]
        concat_size = 768 * 4
        final_size = 768
        self.fulcon = torch.nn.Sequential(torch.nn.Linear(concat_size, int(concat_size*0.8)),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(int(concat_size*0.8), int(concat_size*0.5)),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(int(concat_size*0.5), final_size),
                                          torch.nn.ReLU())
                
        if mtype == 'stl':
            self.output_hot = torch.nn.Linear(final_size, 1)
        elif mtype == 'smythe':
            self.output_soft = torch.nn.Linear(final_size, 2)
        elif mtype == 'mtl':
            self.output_hot = torch.nn.Linear(final_size, 1)
            self.output_soft = torch.nn.Linear(final_size, 2)
            
        self.mtype = mtype
        self.smythe = smythe
            

    def forward(self, text, hypothesis, one_hot_labels, soft_labels, weights, eval=True):
 
        concat_input = torch.cat([text, hypothesis, text*hypothesis, torch.abs(text-hypothesis)], 1)

        ful = self.fulcon(concat_input)
        pred_hot = self.output_hot(ful)
        sigmoid_scores = torch.sigmoid(pred_hot).squeeze(1)
        if eval:
            return None, None, sigmoid_scores
        else:
            hard_loss = torch.nn.BCELoss(reduction='none')(sigmoid_scores, one_hot_labels) * weights
        return hard_loss.sum(), None, sigmoid_scores

"""**Weighting by Item difficulty**"""

item_weights = train_diff

# LEARNING WITH THE GOLD ALONE
NUM_EXPERIMENTS = 30

accs = []
prfs = []
ct_prfs = []
jsds = []
kls = []
similarity_ents = []
ents_correlation = []
ce_results = []

wh_dictionary = {}

for exp in range(NUM_EXPERIMENTS):
    start = 0
    end = 80

    test_hots = []
    test_softs = []
    test_preds = []
    test_entropys = []
    test_distrs = []

    print('\nExperiment %d ###########'%exp)
    for i in range(10):
        print(i,'.....',)
        
        train_x1 = embedded_text[:start] + embedded_text[end:]
        train_x2 = embedded_hypothesis[:start] + embedded_hypothesis[end:]
        train_y1 = train_maj[:start] + train_maj[end:]
        train_y2 = train_soft[:start] + train_soft[end:]
        train_weights = item_weights[:start] + item_weights[end:]

        test_x1 = embedded_text[start:end]
        test_x2 = embedded_hypothesis[start:end]
        test_y1 = train_gold[start:end]
        test_y2 = train_soft[start:end]

        assert len(train_x1) == 720
        assert len(test_x1) == 80

        train_x1, train_x2, train_y1 = torch.tensor(train_x1).float().cuda(), torch.tensor(train_x2).float().cuda(), torch.tensor(train_y1).float().cuda()
        test_x1, test_x2, test_y1 = torch.tensor(test_x1).float().cuda(), torch.tensor(test_x2).float().cuda(), torch.tensor(test_y1).float().cuda()
        train_y2, test_y2 = torch.tensor(train_y2).float().cuda(), torch.tensor(test_y2).float().cuda()
        train_y3 = torch.tensor(train_weights).float().cuda()

        # load embeddings for that dataset
        model = RTE_model('stl').cuda()
        optimizer = torch.optim.Adam(params=[p for p in model.parameters()
                                               if p.requires_grad],
                                       lr=0.0001)

        num_epochs = 20

        for epoch in range(1, num_epochs+1):

            model.train()
            loss, _,_ = model(train_x1, train_x2, train_y1, None, train_y3, False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        _, _,test_pred = model(test_x1, test_x2, test_y1, None, None, True)
        test_hots.extend(test_y1.detach().cpu().tolist())
        test_preds.extend(test_pred.detach().cpu().tolist())
        test_softs.extend(test_y2.detach().cpu().tolist())
        test_entropys.extend(item_entropys[start:end])
        test_distrs.extend(train_distr[start:end])

        start += 80
        end += 80
        
    wh_dictionary[str(exp)] = test_preds

    acc, p, r, f = get_acc_f1(test_hots, test_preds)
    cp, cr, cf = get_ct_f1(test_hots, test_preds, test_distrs)
    preds_probs = [[1-item, item] for item in test_preds]
    jsd, kl = get_jsd_kl_div(test_softs, preds_probs)
    preds_ents = [entropy(p)/entropy(norm) for p in preds_probs]
    ent = cosine_similarity(np.array(test_entropys).reshape(1, 800), np.array(preds_ents).reshape(1, 800))[0][0]
    corr = np.corrcoef(test_entropys, preds_ents)[0][1]
    ce_res = cross_entropy(preds_probs, test_softs)

    accs.append(acc)
    prfs.append([p, r, f])
    ct_prfs.append([cp, cr, cf])
    jsds.append(jsd)
    kls.append(kl)
    similarity_ents.append(ent)
    ents_correlation.append(corr)
    ce_results.append(ce_res)
    
    print(acc, f, cf, jsd, kl, ent, corr, ce_res)
    print('#'*50)

import json

writepath = 'drive/My Drive/Colab Notebooks/Significance_Testing/rte_experiments/'

with open(writepath+'rte_whWeightedMV.jsonlines', 'w') as f:
    json.dump(wh_dictionary, f)

print('MV Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

print('\nMV PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(prfs, 0).tolist()
maxs = ['maximums'] + np.max(prfs, 0).tolist()
mins = ['minimums'] + np.min(prfs, 0).tolist()
stds = ['stds'] + np.std(prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

print('\nMV Crowdtruth PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
stds = ['stds'] + np.std(ct_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

print('\nMV JSD stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

print('\nMV KL stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

print('\nMV entropy similarity stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

print('\nMV entropy correlation stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ents_correlation), np.max(ents_correlation), np.min(ents_correlation), np.std(ents_correlation)))

print('\nMV crossentropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))

"""**Plank**"""

from itertools import combinations

hotsize = 2
matrices = [[0 for i in range(hotsize)] for i in range(hotsize)]

s = 0
for inst in train_distr:
    a = []
    for lab in range(len(inst)):
        lab_count = inst[lab]
        a.extend([lab]*lab_count)
    all_pairs = list(combinations(a, 2))
    s += len(all_pairs)
    for i in range(hotsize):
        for j in range(hotsize):
            matrices[i][j] += all_pairs.count((i, j))

matrices

sum_pairs = np.sum(matrices)
sum_pairs, s

plank_matrices = [[0 for i in range(hotsize)] for i in range(hotsize)]

for i in range(hotsize):
    for j in range(i, hotsize):
        i_j_confusion = (matrices[i][j] + matrices[j][i]) / sum_pairs
        plank_matrices[i][j] = i_j_confusion
        plank_matrices[j][i] = i_j_confusion
    else:
        plank_matrices[i][i] = 0.0 # matrices[i][i] / sum_pairs
plank_matrices = np.array(plank_matrices)

plank_matrices

def get_confusion_weights(softmax_predictions, labels):
    predicted_scores = softmax_predictions.detach().cpu().numpy()
    predictions = np.argmax(predicted_scores, 1)
    labels = torch.argmax(labels, 1).cpu().numpy()
    system_majority = np.stack([predictions, labels], 1)
    weights = 1 - torch.tensor([plank_matrices[i][j] for i,j in system_majority]).cuda()
    return weights

class RTE_model(torch.nn.Module):
    def __init__(self, mtype='stl', smythe='kl'):
        super().__init__()

        # concat_size = embedded_hypothesis[0].shape[0]*4
        # final_size = embedded_hypothesis[0].shape[0]
        concat_size = 768 * 4
        final_size = 768
        self.fulcon = torch.nn.Sequential(torch.nn.Linear(concat_size, int(concat_size*0.8)),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(int(concat_size*0.8), int(concat_size*0.5)),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(int(concat_size*0.5), final_size),
                                          torch.nn.ReLU())
                
        self.output_layer = torch.nn.Linear(final_size, 2)
            

    def forward(self, text, hypothesis, target, soft_labels, eval=True):
 
        concat_input = torch.cat([text, hypothesis, text*hypothesis, torch.abs(text-hypothesis)], 1)

        ful = self.fulcon(concat_input)
        pred_hot = self.output_layer(ful)
        softmax_scores  = torch.softmax(pred_hot, 1)
        if eval:
            return None, None, softmax_scores
        else:
            one_hot_labels = torch.nn.functional.one_hot(target)
            weights = get_confusion_weights(softmax_scores, one_hot_labels)
            cross_entropy = torch.mul(one_hot_labels, softmax_scores.log())
            hard_loss = -torch.sum(cross_entropy)
        return hard_loss.sum(), None, softmax_scores

NUM_EXPERIMENTS = 30

accs = []
prfs = []
ct_prfs = []
jsds = []
kls = []
similarity_ents = []
ents_correlation = []
ce_results = []

plank_dictionary = {}

for exp in range(NUM_EXPERIMENTS):
    start = 0
    end = 80

    test_hots = []
    test_softs = []
    test_preds_hot = []
    test_preds_soft = []
    test_entropys = []
    test_distrs = []

    print('\nExperiment %d ###########'%exp)
    for i in range(10):
        print(i,'.....',)
        
        train_x1 = embedded_text[:start] + embedded_text[end:]
        train_x2 = embedded_hypothesis[:start] + embedded_hypothesis[end:]
        train_y1 = train_gold[:start] + train_gold[end:]
        train_y2 = train_soft[:start] + train_soft[end:]

        test_x1 = embedded_text[start:end]
        test_x2 = embedded_hypothesis[start:end]
        test_y1 = train_gold[start:end]
        test_y2 = train_soft[start:end]

        assert len(train_x1) == 720
        assert len(test_x1) == 80

        train_x1, train_x2, train_y1 = torch.tensor(train_x1).float().cuda(), torch.tensor(train_x2).float().cuda(), torch.tensor(train_y1).long().cuda()
        test_x1, test_x2, test_y1 = torch.tensor(test_x1).float().cuda(), torch.tensor(test_x2).float().cuda(), torch.tensor(test_y1).float().cuda()
        train_y2, test_y2 = torch.tensor(train_y2).float().cuda(), torch.tensor(test_y2).float().cuda()

        # load embeddings for that dataset
        model = RTE_model().cuda()
        optimizer = torch.optim.Adam(params=[p for p in model.parameters()
                                               if p.requires_grad],
                                       lr=0.0001)

        num_epochs = 20

        for epoch in range(1, num_epochs+1):

            model.train()
            loss, _,_ = model(train_x1, train_x2, train_y1, train_y2, False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        _, _,test_pred = model(test_x1, test_x2, test_y1, test_y2)
        test_hots.extend(test_y1.detach().cpu().tolist())
        test_preds_hot.extend(torch.argmax(test_pred, 1).detach().cpu().tolist())
        test_preds_soft.extend(test_pred.detach().cpu().tolist())
        test_softs.extend(test_y2.detach().cpu().tolist())
        test_entropys.extend(item_entropys[start:end])
        test_distrs.extend(train_distr[start:end])

        start += 80
        end += 80
        
    plank_dictionary[str(exp)] = test_preds_soft

    acc, p, r, f = get_acc_f1(test_hots, test_preds_hot)
    cp, cr, cf = get_ct_f1(test_hots, test_preds_hot, test_distrs)
    jsd, kl = get_jsd_kl_div(test_softs, test_preds_soft)
    preds_ents = [entropy(p)/entropy(norm) for p in test_preds_soft]
    ent = cosine_similarity(np.array(test_entropys).reshape(1, 800), np.array(preds_ents).reshape(1, 800))[0][0]
    ce_res = cross_entropy(test_preds_soft, test_softs)
    
    accs.append(acc)
    prfs.append([p, r, f])
    ct_prfs.append([cp, cr, cf])
    jsds.append(jsd)
    kls.append(kl)
    similarity_ents.append(ent)
    ents_correlation.append(corr)
    ce_results.append(ce_res)
    
    print(acc, f, cf, jsd, kl, ent, corr, ce_res)
    print('#'*50)

writepath = 'drive/My Drive/Colab Notebooks/Significance_Testing/rte_experiments/'

with open(writepath+'rte_plankWeightedGold.jsonlines', 'w') as f:
    json.dump(plank_dictionary, f)

print('Gold Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

print('\nGold PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(prfs, 0).tolist()
maxs = ['maximums'] + np.max(prfs, 0).tolist()
mins = ['minimums'] + np.min(prfs, 0).tolist()
stds = ['stds'] + np.std(prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

print('\nGold Crowdtruth PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
stds = ['stds'] + np.std(ct_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

print('\nGold JSD stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

print('\nGold KL stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

print('\nGold entropy similarity stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

print('\nGold entropy correlation stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ents_correlation), np.max(ents_correlation), np.min(ents_correlation), np.std(ents_correlation)))

print('\nGold cross entropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))

