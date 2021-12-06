# -*- coding: utf-8 -*-
"""comCIFAR10_complete.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nW5UlRjWy7XNxoIHMkZLJu2dprsBdZfH
"""

from scipy.spatial import distance
from scipy.special import kl_div
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

"""**Preparing the Data**

**1. Loading and normalizing CIFAR10** (courtesy of
https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html)

We will use torchvision, it’s extremely easy to load CIFAR10.
"""

import torch
import torchvision
import torchvision.transforms as transforms


# EPSILON VALUE TO BE USED THROUGHOUT
EPSILON = 1e-12

"""Then we will do Data Augmentation. Pytorch has built-in functions which can help us perform data augmentation. Only the trainset is augmented.

@alexandra: Because I'm using CIFAR10 test set (CIFAR10H from Peterson  https://github.com/jcpeterson/cifar-10h) as my training and development and only train set can be augmented, I have to load two versions of the CIFAR10H.

I use a subset of CIFAR10H as development. Fixed for now but will randomly select a subset later.
"""

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CIFAR10H_train = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_train)


CIFAR10H_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)


CIFAR10_dev = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""**2. Loading CIFAR10H crowd counts and probs** (from https://github.com/jcpeterson/cifar-10h)"""

DATAPATH = './cifar10_data/'

"""The items in the cifar-10h master file. Since he only used the CIFAR 10 test set, they correspond to the test set from section 1 above

1.   crowd_counts : a \[num_items X num_labels] array, each element in row containing annotations
2.   crowd_std_norm : normalized counts using standard normalization
"""

import numpy as np

crowd_counts = np.load(DATAPATH + 'cifar-10h-master/data/cifar10h-counts.npy')
crowd_stdnorm_softs = np.load(DATAPATH + 'cifar-10h-master/data/cifar10h-probs.npy')
ds_posterior = np.load(DATAPATH + 'ds_posterior.npy')
mace_posterior = np.load(DATAPATH + 'mace_softs.npy')

"""**3. Generating the other items I need to run the experiments**"""

mv_labels = np.argmax(crowd_counts, 1)

def observed_agreement(item_counted_votes):
    """Aggrement is computed using Observed Agreement instead of
    Kappa i.e. (Ao-Ae)/(1-Ae) as expected agreement is not computed
    on a per item basis. TODO confirm with Massimo. Massimo said it
    is so I might need to redo this entire thing.
    """
    # the total number of annotators for that item
    c = sum(item_counted_votes)
    # getting the summ product
    numerator = sum([i*(i-1) for i in item_counted_votes])
    if c == 1:
        return 1.0  # if only one annotator annotated it, it is a perfect agreement
    return numerator/(c*(c-1))

crowd_counts_list = crowd_counts.tolist()

items_oas = [observed_agreement(item) for item in crowd_counts_list]

np.array(items_oas[:10])

crowd_counts[9]

print(f'Average observed agreement of cifar10h as released is {np.average(items_oas)}')

from scipy.special import softmax

crowd_softmax_softs = softmax(crowd_counts, 1).round(8)

crowd_softmax_softs[0]

crowd_counts[0]

"""**4. Putting the entire dataset together**"""

SIZE_CIFAR10H = len(CIFAR10H_train)
NUM_CLASSES = 10

train_indices = np.load(DATAPATH+'train_indices.npy').tolist()

test_indices = np.load(DATAPATH+'test_indices.npy').tolist()

train_indices[:5], len(train_indices)

"""**The Train Set**"""

train_images = np.array([CIFAR10H_train[i][0].tolist() for i in range(SIZE_CIFAR10H)])[train_indices]
train_labels = np.array([CIFAR10H_train[i][1] for i in range(SIZE_CIFAR10H)])[train_indices]

train_distr = crowd_counts[train_indices]
train_stdnorm_soft = crowd_stdnorm_softs[train_indices]
train_sm_soft = crowd_softmax_softs[train_indices]
train_ds_soft = ds_posterior[train_indices]
train_mace_soft = mace_posterior[train_indices]
train_oas = np.array(items_oas)[train_indices]

print(train_images.shape, train_labels.shape, train_distr.shape)

# this was the point of all that round about shuffling
[train_labels.tolist().count(i) for i in range(NUM_CLASSES)]

"""**The Test Set**"""

test_images = np.array([CIFAR10H_test[i][0].tolist() for i in range(SIZE_CIFAR10H)])[test_indices]
test_labels = np.array([CIFAR10H_test[i][1] for i in range(SIZE_CIFAR10H)])[test_indices]

test_distr = crowd_counts[test_indices]
test_stdnorm_soft = crowd_stdnorm_softs[test_indices]
test_sm_soft = crowd_softmax_softs[test_indices]

print(test_images.shape, test_labels.shape, test_distr.shape)

NUM_CLASSES = 10
norm = [1/NUM_CLASSES for i in range(NUM_CLASSES)]
#test_entropys = [entropy(soft)/entropy(norm) for soft in test_sm_soft]
test_entropys = [entropy(soft)/entropy(norm) for soft in test_stdnorm_soft]

"""**The Development Set**"""

dev_images = np.array([CIFAR10_dev[i][0].tolist() for i in range(len(CIFAR10_dev))])
dev_labels = np.array([CIFAR10_dev[i][1] for i in range(len(CIFAR10_dev))])

print(dev_images.shape, dev_labels.shape)

"""**Also, MV aggregated train and test sets**"""

train_mv = np.argmax(train_distr, 1)
(train_mv == train_labels).sum() / len(train_mv)

test_mv = np.argmax(test_distr, 1)

"""**Evaluation Metrics**"""

def get_acc_f1(test_trues, test_preds,num_classes=NUM_CLASSES):
    total = 0
    correct = 0

    matches = {i:0 for i in range(num_classes)}
    gold = {i:0 for i in range(num_classes)}
    system = {i:0 for i in range(num_classes)}

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

def get_ct_f1(test_trues, test_preds, test_distrs, num_classes=NUM_CLASSES):
    tp = {i:0 for i in range(num_classes)}
    fp = {i:0 for i in range(num_classes)}
    fn = {i:0 for i in range(num_classes)}

    gold = {i:0 for i in range(num_classes)}


    for p, g, distr in zip(test_preds, test_trues, test_distrs):
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

norm = [1/NUM_CLASSES for i in range(NUM_CLASSES)]

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


"""**Define a Convolution Neural Network**"""

import torch.nn as nn
import torch.nn.functional as F

def get_ce_loss(scores, labels_long):
    '''Args:
        true_labels: either one hot labels or a probability distribution
        predicted_labels: output of the model
    '''
    true_labels = torch.nn.functional.one_hot(labels_long, NUM_CLASSES).float()
    scores = nn.LogSoftmax(dim=1)(scores)
    cross_entropy = torch.mul(true_labels, scores)
    loss = -torch.sum(cross_entropy)
    return loss


class ResNet(nn.Module):

    def __init__(self, n=7, res_option='A', use_dropout=False):
        super(ResNet, self).__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(n, 16, 16, 1)
        self.layers2 = self._make_layer(n, 32, 16, 2)
        self.layers3 = self._make_layer(n, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 10)
        # for the mtl
        self.linear_2 = nn.Linear(64, 1)

    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride, res_option=self.res_option, use_dropout=self.use_dropout),
            *[ResBlock(channels) for _ in range(layer_count-1)])

    def forward(self, x, hard_labels=None, items_weights=None, is_eval=True):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
       
        out_1 = self.linear(out)
        out_2 = self.linear_2(out)
        if is_eval:
            return out_1
        else:
            hard_loss = get_ce_loss(out_1, hard_labels)

            pred_items_diff = torch.sigmoid(out_2).squeeze(1)
            soft_loss = torch.nn.MSELoss(reduction='sum')(items_weights, pred_items_diff)
            return hard_loss, soft_loss


class ResBlock(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            if res_option == 'A':
                self.projection = IdentityPadding(num_filters, channels_in, stride)
            elif res_option == 'B':
                self.projection = ConvProjection(num_filters, channels_in, stride)
            elif res_option == 'C':
                self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu2(out)
        return out


# various projection options to change number of filters in residual connection
# option A from paper
class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

# option B from paper
class ConvProjection(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv2d(channels_in, num_filters, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        return out

# experimental option C
class AvgPoolPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool2d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

"""**GETTING THE VALIDATION AND TEST DATA**"""

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

bs = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_ds = TensorDataset(torch.tensor(dev_images).float().to(device), torch.tensor(dev_labels).long().to(device))
test_ds = TensorDataset(torch.tensor(test_images).float().to(device), torch.tensor(test_labels).long().to(device))

val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=False)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=False)

def predict_using_model(model, dataloader):
    pred_probs = []
    pred_labels = []
    target_labels = []
    for batch in dataloader:
        x, y = batch[0], batch[1]

        with torch.no_grad():
            preds = model(x)
            preds = torch.softmax(preds, 1).detach().cpu().numpy()
            pred_labels.extend(np.argmax(preds, 1).tolist())
            pred_probs.extend(preds.tolist())

        target_labels.extend(y.detach().cpu().numpy().tolist())

    return target_labels, pred_probs, pred_labels


def check_accuracy(model, loader, num_classes=NUM_CLASSES):
    total = 0
    correct = 0

    num_correct = 0
    num_samples = 0

    matches = {i:0 for i in range(num_classes)}
    gold = {i:0 for i in range(num_classes)}
    system = {i:0 for i in range(num_classes)}

    model.eval()
    for X, y in loader:
        X_var = Variable(X).to(device)
        y = y.cpu()
        scores = model(X_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

        for p, g in zip(preds.tolist(), y.tolist()):
            total+=1
            if p == g:
                correct+=1
                matches[p] += 1

            gold[g] += 1
            system[p] += 1

    acc = float(num_correct) / num_samples

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

    acc2 = correct/total

    print('Got %d / %d correct (%.2f) (%.2f), f1: (%.2f)' % (num_correct, num_samples, 100 * acc, 100 * acc, 100 * average_f1))

    return acc, average_precision, average_recall, average_f1


def get_param_count(model):
    param_counts = [np.prod(p.size()) for p in model.parameters()]
    return sum(param_counts)

"""**TRAINING**"""
print('###################################Training Using Soft Labels ##########################################')

"""**Soft Labelling**"""

def train(loader_train, model, criterion, optimizer):
    model.train()
    for t, (X, yhard, ysoft) in enumerate(loader_train):
        X_var = Variable(X)
        y_var1 = Variable(yhard)
        y_var2 = Variable(ysoft)
        hard_loss, soft_loss  = model(X_var, y_var1, y_var2, False)

        if (t+1) % print_every == 0:
            print('t = %d, loss = %.4f' % (t+1, hard_loss.data))

        optimizer.zero_grad()
        soft_loss.backward(retain_graph=True)
        hard_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


import os
from torch.autograd import Variable
import torch.optim as optim


torch.cuda.empty_cache()


NUM_TEST = 3000

NUM_EXPERIMENTS = 10

accs = []
prfs = []
ct_prfs = []
jsds = []
kls = []
similarity_ents = []
ents_correlation = []
ce_results = []

print_every = 100

mtl_uma_dictionary = {}

for i in range(NUM_EXPERIMENTS):
    best_val_f1 = 0

    trn_ds = TensorDataset(torch.tensor(train_images).float().to(device), torch.tensor(train_labels).long().to(device),
                           torch.tensor(train_oas).float().to(device))
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, pin_memory=False)

    model = ResNet(res_option='A')
    gpu_dtype = torch.cuda.FloatTensor
    model = model.type(gpu_dtype)

    param_count = get_param_count(model)
    print('Parameter count: %d' % param_count)


    learning_rate = 0.1
    weight_decay = 0.0001

    SCHEDULE_EPOCHS = [50, 5, 10] # divide lr by 10 after each number of epochs

    for num_epochs in SCHEDULE_EPOCHS:
        print('\nTraining for %d epochs with learning rate %f' % (num_epochs, learning_rate))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            print('Starting epoch %d / %d' % (epoch+1, num_epochs))
            train(trn_dl, model, None, optimizer)
            dev_accuracy,_,_,dev_f1 = check_accuracy(model, val_dl)

            if dev_f1 > best_val_f1:
                best_val_f1 = dev_f1
                print('saving best model')
                torch.save(model.state_dict(), DATAPATH+'best_dssoftmodel.pt')

        learning_rate *= 0.1

    model = ResNet(res_option='A').to(device)
    model.load_state_dict(torch.load(DATAPATH+'best_dssoftmodel.pt'))

    model.eval()
    _, test_pred, predicted_test_labels = predict_using_model(model, test_dl)

    mtl_uma_dictionary[str(i)] = test_pred

    test_acc, test_p, test_r, test_f = get_acc_f1(test_labels, predicted_test_labels)
    cp, cr, cf = get_ct_f1(test_labels.tolist(), predicted_test_labels, test_distr)

    jsd, kl = get_jsd_kl_div(test_stdnorm_soft, test_pred)

    preds_ents = [entropy(p)/entropy(norm) for p in test_pred]

    ent = cosine_similarity(np.array(test_entropys).reshape(1, NUM_TEST), np.array(preds_ents).reshape(1, NUM_TEST))[0][0]

    corr = np.corrcoef(test_entropys, preds_ents)[0][1]
    
    ce_res = cross_entropy(test_pred, test_stdnorm_soft)


    accs.append(test_acc)
    prfs.append([test_p, test_r, test_f])
    ct_prfs.append([cp, cr, cf])
    jsds.append(jsd)
    kls.append(kl)
    similarity_ents.append(ent)
    ents_correlation.append(corr)
    ce_results.append(ce_res)

    print(test_acc, test_f, cf, jsd, kl, ent, corr, ce_res)
    print('#'*50)

import json
writepath = './predictions/'
name = 'mtl_oa'

with open(writepath+'cifar_' + name + '.jsonlines', 'w') as f:
    json.dump(mtl_uma_dictionary, f)


print('MTL_UMA Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

print('\nMTL_UMA PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(prfs, 0).tolist()
maxs = ['maximums'] + np.max(prfs, 0).tolist()
mins = ['minimums'] + np.min(prfs, 0).tolist()
stds = ['stds'] + np.std(prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

print('\nMTL_UMA Crowdtruth PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
stds = ['stds'] + np.std(ct_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

print('\nMTL_UMA JSD stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

print('\nMTL_UMA KL stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

print('\nMTL_UMA entropy similarity stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

print('\nMTL_UMA entropy correlation stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ents_correlation), np.max(ents_correlation), np.min(ents_correlation), np.std(ents_correlation)))


print('\nMTL_UMA cross entropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))


