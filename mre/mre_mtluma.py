# -*- coding: utf-8 -*-
"""softmax_weighting_mre.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TFSh8gK5pKFhg-ywg0--PSYH3SLnJUIC

# **1. Introduction (You can't skip the basics Alexandra)**
"""
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.'%torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead')
    device = torch.device('cpu')



import os
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

"""###**1.1. Evaluation metrics**"""

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


"""**2. Loading the Dataset from Crowd Truth**"""
DATAPATH = './mre_data'

all_data = []
with open(DATAPATH+"/ground_truth_cause.csv",encoding='utf8' ) as fd:
    rd = csv.reader(fd, delimiter=",")
    for row in rd:
        if row != []:
            all_data.append(row)

# using only the items with expert labels
data_dict = {}
for sid, _, srs, crwd, _, expert, _, _,b1, e1, _, b2, e2, sent,_,_,_ in all_data[1:]:
    if expert != 'NA':
        data_dict[sid] = [sent, float(srs), (b1, e1), (b2, e2), (crwd, expert)]

assert len(data_dict) == 975

print('A sample of the dataset:')
data_dict['820001']

# get annotator data for the 975 sentences
the975 = data_dict.keys()

annotator_dict = {}
filepath = DATAPATH+'/RelEx/'
for raw_batch in os.listdir(DATAPATH+'/RelEx'):
    batch_path = filepath + raw_batch
    batch_data = []
    with open(batch_path, encoding='utf8' ) as fd:
        rd = csv.reader(fd, delimiter=",")
        for row in rd:
            if row != []:
                batch_data.append(row)
    for item in batch_data[0:]:
        wkr_id, relations, sent_id = item[7], item[12], item[22].split('-')[0]
        if sent_id in the975:
            if sent_id not in annotator_dict:
                annotator_dict[sent_id] = {}
            # each workers annotation for a given sentences is collated into one occurence
            annotator_dict[sent_id][wkr_id] = relations

assert len(annotator_dict) == 975

lens = [len(annotator_dict[sid]) for sid in the975]

print(lens[0], len(set(lens)))

set(lens)

annotator_dict['820001']

for sent in data_dict:
    anns = annotator_dict[sent]
    # for majority voting , against and for
    crowd_counts = [0, 0] # [num that didn't say causes, num that said causes]
    for ann, relns in anns.items():
        if '[CAUSES]' in relns:
            crowd_counts[1] += 1
        else:
            crowd_counts[0] += 1
    data_dict[sent].append(crowd_counts)

data_dict['820001']

# the dawid and skene dataset
ds_dict = np.load(DATAPATH + '/ds_results.npy', allow_pickle=True).item()

"""**Training Data Preparation**"""

from scipy.special import softmax

# creating the train and development data.
train_sents = []
train_experts = []
train_crowd = []
train_distr = []
train_maj = []
train_soft = []
ds_softs = []
ds_labels = []
srs_scores = []

for sent in data_dict:
    sent_info = data_dict[sent]
    train_sents.append(sent_info[0])
    exp = float(sent_info[-2][1])
    if exp == -1:
        exp = [1, 0]
    else:
        exp = [0, 1]
    train_experts.append(exp)
    crwd = float(sent_info[-2][0])
    if sent_info[1] >= 0.7 :
        crwd = [0, 1]
    else:
        crwd = [1, 0]
    train_crowd.append(crwd)
    maj_dummy = [0, 0]
    maj_dummy[np.argmax(sent_info[-1])] = 1
    train_maj.append(maj_dummy)
    distr = sent_info[-1]
    num_votes = sum(distr)
    train_distr.append(distr)
    train_soft.append([i/num_votes for i in distr])

    sent_ds = ds_dict[sent]
    ds_dummy = [0,0]
    ds_dummy[np.argmax(sent_ds)] = 1
    ds_labels.append(ds_dummy)
    ds_softs.append(sent_ds)

    srs_scores.append(sent_info[1])

def observed_agreement(counted_votes):
    """Aggrement is computed using Observed Agreement instead of
    Kappa i.e. (Ao-Ae)/(1-Ae) as expected agreement is not computed
    on a per item basis. TODO confirm with Massimo.
    """
    c = sum(counted_votes)
    # getting the summ product
    numerator = sum([i*(i-1) for i in counted_votes])
    if c == 1:
        return 1.0  # if only one annotator annotated it, it is a perfect agreement
    return numerator/(c*(c-1))

item_oas = [observed_agreement(v) for v in train_distr]

norm = [0.5, 0.5]
item_entropys = [entropy(scores)/entropy(norm) for scores in train_soft]

item_srsscores = []

"""# **3. Tokenization & Input Formatting**

Transforming the data into a form that BERT can train on.

### **3.1. BERT Tokenizer**

To feed the text into BERT, it must be split into tokens, and these tokens must be mapped to their index in the tokenizer vocabulary.

The tokenization must be performed by the tokenizer included with BERT. We'll be using the 'uncased' version of BERT here.
"""

import transformers

# bert_path = DATAPATH + '/biobert_v1.1_pubmed'

from transformers import BertTokenizer

# Load the BERT tokenizer
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)

"""Applying the tokenizer to one sentence just to see the output"""

# Print the original sentence
print('Original: ', train_sents[0])

# Print the sentence split into tokens
print('Tokenized: ', tokenizer.tokenize(train_sents[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_sents[0])))

"""tokenize.encode function handles both steps, tokenize and convert_tokens_to_ids

### **3.3. Tokenize Dataset**

The transformers library provides a helpful encode function which will handle most of the parsing and data prep steps for us.

Before we are ready to encode our text, though, we need to decide on a **maximum sentence length** for padding/truncating to.

The below cell will perform one tokenization pass of the dataset in order to measure the maximum sentence length.
"""

max_len = 0
min_len = 10000
for sent in train_sents:
    # Tokenizes the text and adds '[CLS]' and '[SEP]' tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
    min_len = min(min_len, len(input_ids))
print('Max sentence length: ', max_len)
print('Min len:', min_len)

lss = [len(sent) for sent in train_sents]

np.average(lss)

input_ids = []
attention_masks = []
max_len = 200

for sent in train_sents:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens = True,
        max_length=max_len,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# convert the list of tensors into tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Print sentence 0, now a tensor of IDs
print('Original', train_sents[0])
print('Token IDs:', input_ids[0])

"""#**4. Training Our Classification Model**

Now that our input data is properly formatted, it's time to fine tune the BERT model.
"""

import random
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from torch.utils.data import TensorDataset



class MyBertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.auxiliary_classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, item_weights=None):
        outputs = self.bert(input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        softmax_scores = torch.softmax(logits, 1)
        outputs = (softmax_scores,) + outputs[2:]
        cross_entropy = torch.mul(labels, softmax_scores.log())
        loss = -torch.sum(cross_entropy)

        auxiliary_logits = self.auxiliary_classifier(pooled_output)
        pred_items_diff = torch.sigmoid(auxiliary_logits).squeeze(1)
        auxiliary_loss = torch.nn.MSELoss(reduction='sum')(item_weights, pred_items_diff)

        outputs = (loss,) + (auxiliary_loss,) + outputs
        return outputs # (loss), logits, ((hidden_states), (attentions))


"""**MTL UMA USING GOLD LABELS**"""
labels = torch.tensor(train_experts)
torch.cuda.empty_cache()

# the necessary setup
epochs = 4
num_experiments = 10
#----------- val size for a 5-fold cross-validation
val_size = len(labels) / 5


# the metrics to be collected, accross all the experiments
prfs = []
ct_prfs = []
accs = []
jsds = []
kls = []
similarity_ents = []
ents_correlations = []
ce_results = []

prediction_dictionary = {}
ids = np.load(DATAPATH+'/randomized_ids.npy')

# running the experiments
for experiment in range(num_experiments):

    print('=======================EXPERIMENT %d ======================'%(experiment+1))

    # five fold cross validation
    start = 0
    end = 195


    # the results for the entire dataset accross the folds
    total_eval_acc = 0.0
    total_eval_cprf = np.array([0.0, 0.0, 0.0])
    total_eval_prf = np.array([0.0, 0.0, 0.0])
    total_eval_jsd = 0.0
    total_eval_kl = 0.0
    total_eval_sim = 0.0
    total_eval_corr = 0.0
    total_eval_ce = 0.0

    experiment_predictions = []

    # a different shuffle for each experiment
    input_ids_, attention_masks_, labels_ = input_ids[ids], attention_masks[ids], labels[ids]
    input_ids_, attention_masks_, labels_ = input_ids_.cpu().tolist(), attention_masks_.cpu().tolist(), labels_.cpu().tolist()


    # other things I need
    soft_labels_ = (np.array(train_soft)[ids]).tolist()
    item_weights_ = (np.array(item_oas)[ids]).tolist()

    train_experts_ = np.array(train_experts)
    gold_labels = train_experts_[ids].tolist()

    train_distr_ = np.array(train_distr)
    the_distrs = train_distr_[ids].tolist()

    item_entropys_ = np.array(item_entropys)
    the_entropys = item_entropys_[ids].tolist()

    # training using 5-fold cross validation
    for split_i in range(0, 5):

        torch.cuda.empty_cache()
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classfication layer on top.
        MODEL_TYPE = 'stl_hardCE'
        model = MyBertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels = 2,
                    output_attentions = False,
                    output_hidden_states = False
            )

        # Tell pytorch to run this model on the GPU
        model.cuda()

        optimizer = AdamW(model.parameters(),
                        lr = 2e-5,
                        eps = 1e-8
                        )

        print("")
        print('===== Fold {:} =========='.format(split_i + 1))
        print('Training...')

        train_input_ids = input_ids_[:start] + input_ids_[end:]
        train_attn_masks = attention_masks_[:start] + attention_masks_[end:]
        train_labels = labels_[:start] + labels_[end:]
        train_weights = item_weights_[:start] + item_weights_[end:]

        val_input_ids = input_ids_[start:end]
        val_attn_masks = attention_masks_[start:end]
        val_weights = item_weights_[start:end]

        val_labels = gold_labels[start:end]
        val_distr = the_distrs[start:end]
        val_softs = soft_labels_[start:end]
        val_entropys = the_entropys[start:end]

        assert len(train_input_ids) == 780
        assert len(val_input_ids) == 195

        train_x1, train_x2, train_y1 = torch.tensor(train_input_ids).long().cuda(), torch.tensor(train_attn_masks).long().cuda(), torch.tensor(train_labels).float().cuda()
        test_x1, test_x2, test_y1 = torch.tensor(val_input_ids).long().cuda(), torch.tensor(val_attn_masks).long().cuda(), torch.tensor(val_labels).float().cuda()

        # test_y2 doesn't really matter, it's just lazy programming.
        train_y2, test_y2 = torch.tensor(train_weights).float().cuda(), torch.tensor(val_entropys).float().cuda()

        # for fine-turning BERT, the authors recommend a batch size of 16 or 32
        batch_size = 16

        # creating and batching the dataset
        train_dataset = TensorDataset(train_x1, train_x2, train_y1, train_y2)
        val_dataset = TensorDataset(test_x1, test_x2, test_y1, test_y2)

        # Create the DataLoaders for our training and validation sets
        train_dataloader = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset), # select batches randomly
                        batch_size = batch_size
        )

        # For validation, order doesn't matter so we'll just read them sequentially
        validation_dataloader = DataLoader(
                            val_dataset,
                            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially
                            batch_size = batch_size
        )


        # Total number of training steps is [number of batches] X [number of epochs]
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = 0, # default value in run_glue.py
                                                    num_training_steps = total_steps)

        # train each batch for the required number of epochs
        for epoch_i in range(0, epochs):

            # ===============================================
            #                Training
            #================================================

            # Perform one full pass over the training set
            print("")
            print('===== Epoch {:} / {:} =========='.format(epoch_i + 1, epochs))

            model.train()

            # In each epoch, batch the dataset and train on all of it
            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_weights = batch[3].to(device)

                model.zero_grad()
                loss, auxiliary_loss, logits = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    item_weights=b_weights)

                auxiliary_loss.backward(retain_graph=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        print('Done training batch %d'%(split_i + 1))
        #===================================================
        #       Testing for that batch
        #===================================================
        print("")
        print("Running Validation for batch %d"%(split_i + 1))
        pred_probs = []
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_weights = batch[3].to(device)

            with torch.no_grad():
                loss, _, logits = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    item_weights=b_weights)

            logits = logits.detach().cpu().numpy()
            pred_probs.extend(logits.tolist())

        pred_probs = np.array(pred_probs)
        pred_flat = np.argmax(pred_probs, axis=1)
        label_ids = np.argmax(val_labels, 1)

        experiment_predictions.extend(pred_probs.tolist())

        fold_eval_acc, fold_p,fold_r,fold_f = get_acc_f1(label_ids, pred_flat)
        fold_cp, fold_cr, fold_cf = get_ct_f1(label_ids, pred_flat, val_distr)
        fold_jsd, fold_kl = get_jsd_kl_div(val_softs, pred_probs)
        fold_pred_ents = [entropy(p)/entropy(norm) for p in pred_probs]
        fold_ent_sim = cosine_similarity(np.array(val_entropys).reshape(1, 195), np.array(fold_pred_ents).reshape(1, 195))[0][0]
        fold_corr = np.corrcoef(val_entropys, fold_pred_ents)[0][1]
        fold_ce = cross_entropy(pred_probs, val_softs)


        total_eval_acc += fold_eval_acc
        total_eval_prf += np.array([fold_p, fold_r, fold_f])
        total_eval_cprf += np.array([fold_cp, fold_cr, fold_cf])
        total_eval_jsd += fold_jsd
        total_eval_kl += fold_kl
        total_eval_sim += fold_ent_sim
        total_eval_corr += fold_corr
        total_eval_ce += fold_ce

        print('Fold %d and f1:%0.4f, %0.4f'%(split_i+1, fold_eval_acc, fold_f))

        start += 195
        end += 195

    # the experiment results are the averages across the 5 fold for each metric
    avg_val_acc = total_eval_acc / 5
    avg_val_prf = total_eval_prf / 5
    avg_val_cprf = total_eval_cprf / 5
    avg_val_jsd = total_eval_jsd / 5
    avg_val_kl = total_eval_kl / 5
    avg_val_sim = total_eval_sim / 5
    avg_val_corr = total_eval_corr / 5
    avg_ce_res = total_eval_ce / 5

    prediction_dictionary[str(experiment)] = experiment_predictions

    accs.append(avg_val_acc)
    prfs.append(avg_val_prf.tolist())
    ct_prfs.append(avg_val_cprf.tolist())
    jsds.append(avg_val_jsd)
    kls.append(avg_val_kl)
    similarity_ents.append(avg_val_sim)
    ents_correlations.append(avg_val_corr)
    ce_results.append(avg_ce_res)

    print('Accuracy: {0:.4f}, F1: {1:.4f}, CT F1:{2:.4f}'.format(avg_val_acc, avg_val_prf[2], avg_val_cprf[2] ))
    print('JSD: {0:.4f}, ENT_SIM: {1:.4f}, Correlation: {2:.4f}'.format(avg_val_jsd, avg_val_sim, avg_val_corr))
    print('Cross Entropy: {0:.4f}'.format(avg_ce_res))

    print('#'*50)


import json
writepath = './predictions/'

name = 'mtl_oa'
with open(writepath+ 'mre_' + name + '.jsonlines', 'w') as f:
    json.dump(prediction_dictionary, f)


print(name.upper() + ' Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

print('\n' + name.upper() + ' PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(prfs, 0).tolist()
maxs = ['maximums'] + np.max(prfs, 0).tolist()
mins = ['minimums'] + np.min(prfs, 0).tolist()
stds = ['stds'] + np.std(prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

print('\n' + name.upper() + ' Crowdtruth PRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
stds = ['stds'] + np.std(ct_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

print('\n' + name.upper() + ' JSD stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

print('\n' + name.upper() + ' KL stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

print('\n' + name.upper() + ' entropy similarity stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

print('\n' + name.upper() + ' entropy correlation stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ents_correlations), np.max(ents_correlations), np.min(ents_correlations), np.std(ents_correlations)))

print('\n' + name.upper() + ' cross entropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))

