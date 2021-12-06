# -*- coding: utf-8 -*-
"""softmax_pos_ct.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iXpiBv0A7_1LyDix4_VMJnWpS_HI9w1v
"""

from google.colab import drive

drive.mount('/content/drive')

import torch
import time
import numpy as np
import torch.utils.data as data_utils
import pickle

from scipy.spatial import distance
from scipy.special import kl_div
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def readbin(f_in):
    inp = open(f_in, "rb")
    out = pickle.load(inp)
    inp.close()
    return out

DATA_PATH = 'drive/My Drive/Data/pos_data'

word_emb_matrix = DATA_PATH + '/word/dh_emb_matrix.bin'
word_pad_matrix = DATA_PATH + '/word/dh_pad_csr.bin'
word_id_in_sent  = DATA_PATH + '/word/dh_i_word_in_sent.bin'

char_emb_matrix = DATA_PATH + '/char/dh_emb_matrix.bin'
char_pad_matrix = DATA_PATH + '/char/dh_pad_csr.bin'

hot_path = DATA_PATH + '/labels/dh_hot_csr.bin'
soft_devtrn_path = DATA_PATH + '/raw_soft_labels/soft_devtrn.bin'
soft_tst_path = DATA_PATH + '/raw_soft_labels/soft_tst.bin'

softtstsize = 7877
tstsize = 3064
devsize = 2439

lstm_size = 128
attn_size = 512

num_epochs = 20

batsize = 1000
sizeout_rate = 0.8

word_emb_size = 300
char_emb_size = 64

num_users = 178
train_users = DATA_PATH + '/train_anno_users_pg.npy'
train_status = DATA_PATH + '/train_anno_status_pg.npy'

word_emb = readbin(word_emb_matrix)
word_iis = readbin(word_id_in_sent)
word_pad = readbin(word_pad_matrix)
char_emb = readbin(char_emb_matrix)
char_pad = readbin(char_pad_matrix)
soft_d_t = readbin(soft_devtrn_path)
soft_tst = readbin(soft_tst_path)

hot  = readbin(hot_path)
hotsize = hot.shape[1]
word_padsize = word_pad.shape[1]
char_padsize = char_pad.shape[1]

word_embedding_lookup = torch.from_numpy(word_emb).float().to(device)
char_embedding_lookup = torch.from_numpy(char_emb).float().to(device)

softst_tst  = softtstsize + tstsize
softst_tst_dev = softtstsize + tstsize + devsize

word_iis_tst  = word_iis[softtstsize:softst_tst] # da lowlands.test.tsv
word_iis_dev  = word_iis[softst_tst:softst_tst_dev] # da gimpel_crowdsourced
word_iis_trn  = word_iis[softst_tst_dev:] # da gimpel_crowdsourced

word_pad_tst  = word_pad[softtstsize:softst_tst].toarray() # da lowlands.test.tsv
word_pad_dev  = word_pad[softst_tst:softst_tst_dev].toarray() # da gimpel_crowdsourced
word_pad_trn  = word_pad[softst_tst_dev:].toarray() # da gimpel_crowdsourced

char_pad_tst  = char_pad[softtstsize:softst_tst].toarray() # da lowlands.test.tsv
char_pad_dev  = char_pad[softst_tst:softst_tst_dev].toarray() # da gimpel_crowdsourced
char_pad_trn  = char_pad[softst_tst_dev:].toarray() # da gimpel_crowdsourced

hot_tst = hot[:tstsize].toarray() # da lowlands.test.tsv
hot_dev = hot[tstsize:tstsize+devsize].toarray() # da gimpel_crowdsourced
hot_trn = hot[tstsize+devsize:].toarray() # da gimpel_crowdsourced

soft_test = soft_tst.values 
soft_dev = soft_d_t[:devsize].values # da gimpel_crowdsourced
soft_train = soft_d_t[devsize:].values # da gimpel_crowdsourced

def lookup_embeddings(embedding_lookup, index_matrix):
    flattened_indices = torch.flatten(index_matrix)
    selected = torch.index_select(embedding_lookup, 0, flattened_indices)
    return selected.reshape(index_matrix.shape[0], index_matrix.shape[1], embedding_lookup.shape[-1])


def to_numpy(torch_tensor):
    return torch_tensor.cpu().clone().detach().numpy()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def log2(x):
    numerator = torch.log(x)
    denom = to_cuda(torch.log(torch.tensor([2]).float()))
    out = numerator / denom
    out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    return out

def create_dataset(word_pad_trn_bat, word_iis_trn_bat, char_pad_trn_bat, y_hot_trn_bat, y_soft_trn_bat):
    word_pad_trn_bat, word_iis_trn_bat = torch.from_numpy(word_pad_trn_bat).long().to(device), torch.from_numpy(word_iis_trn_bat).long().to(device)
    char_pad_trn_bat, y_hot_trn_bat = torch.from_numpy(char_pad_trn_bat).long().to(device), torch.from_numpy(y_hot_trn_bat).float().to(device)
    y_soft_trn_bat = torch.from_numpy(y_soft_trn_bat).float().to(device)
    return word_pad_trn_bat, word_iis_trn_bat, char_pad_trn_bat, y_hot_trn_bat, y_soft_trn_bat


def backprop_hot(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

def backprop_both(optimizer, loss1, loss2):
    optimizer.zero_grad()
    loss2.backward(retain_graph=True)
    loss1.backward()
    optimizer.step()
    return

def get_acc_f1(test_trues, test_preds, num_classes=hotsize):
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

def get_ct_f1(test_trues, test_preds, test_distrs, num_classes=hotsize):

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

def get_jsd_kl_div(soft_probs, predicted_probs):
    num_items = len(predicted_probs)
    all_jsd = [distance.jensenshannon(soft_probs[i], predicted_probs[i]) for i in range(num_items)]
    all_kl = [kl_div(soft_probs[i], predicted_probs[i]) for i in range(num_items)]
    return np.sum(all_jsd)/num_items, np.sum(all_kl)/num_items

def get_predictions(model, eval_loader, mtl=False):
    hard_preds = []
    soft_preds = []
    model.eval()
    for wwordpad_bat, wwordiis_bat, ccharpad_bat, hhot_bat, ssoft_bat in eval_loader:
        if not mtl:
            one_hot_pred, _,_,_ = model(wwordpad_bat, wwordiis_bat, ccharpad_bat, hhot_bat, ssoft_bat)
            one_hot_pred = one_hot_pred.detach().cpu().numpy()
            # the soft and hard predictions come from the same dense layer in the single task models.
            hard_preds.extend(np.argmax(one_hot_pred, 1))
            soft_preds.extend(one_hot_pred)
        else:
            one_hot_pred, soft_pred, _, _ = model(wwordpad_bat, wwordiis_bat, ccharpad_bat, hhot_bat, ssoft_bat)
            hard_preds.extend(one_hot_pred.argmax(-1).detach().cpu().numpy())
            soft_preds.extend(soft_pred.detach().cpu().numpy())
    return hard_preds, soft_preds

"""**Geting, Tensorizing and Batching the Data**"""

from scipy.special import softmax

NUM_TEST = len(hot_dev)
norm = [1/hotsize for i in range(hotsize)]
test_entropys = []
test_softs = []
test_distr = []
for distr in soft_dev:
    test_distr.append(distr)
    num_votes = np.sum(distr)
    soft = softmax(distr)
    test_softs.append(soft)
    ent = entropy(soft)/entropy(norm)
    test_entropys.append(ent)

test_softs = np.array(test_softs)
test_entropys = np.array(test_entropys)
test_distr = np.array(test_distr)
test_softs.shape

train_maj = []
train_softs = []
train_distr = []
for distr in soft_train:
    train_distr.append(distr)
    num_votes = np.sum(distr)
    soft = [i/num_votes for i in distr]
    train_softs.append(soft)
    train_maj.append(np.argmax(distr))

N_CLASSES = hotsize
train_srs = []
for item in train_distr:
    item = np.array(item).reshape(1, N_CLASSES)
    srs_vector = []
    for label in range(N_CLASSES):
        unit_vector = np.array([0] * N_CLASSES).reshape(1, N_CLASSES)
        unit_vector[0][label] = 1
        srs = cosine_similarity(item, unit_vector)
        srs_vector.append(srs[0][0])
    # print(item, srs_vector)
    train_srs.append(srs_vector)

np.array(train_srs[:4])

train_softs = np.array(train_softs)
train_distr = np.array(train_distr)
train_srs = np.array(train_srs).round(6)
hot_trn_maj = np.eye(hotsize)[train_maj]
print(hot_trn_maj.shape)

# Alexandra, for these experiments, dev data for testing and test data for validation because dev data has actual soft labels.
word_pad_test_tens, word_iis_test_tens, char_pad_test_tens, hot_test_tens, soft_test_tens = create_dataset(word_pad_tst, word_iis_tst, char_pad_tst, hot_tst, soft_test)
dev = data_utils.TensorDataset(word_pad_test_tens, word_iis_test_tens, char_pad_test_tens, hot_test_tens, soft_test_tens[:len(hot_test_tens)]) # the soft test is just a placeholder and not actually useful or used
dev_loader = data_utils.DataLoader(dev, batch_size=batsize, shuffle=False)

word_pad_dev_tens, word_iis_dev_tens, char_pad_dev_tens, hot_dev_tens, soft_dev_tens = create_dataset(word_pad_dev, word_iis_dev, char_pad_dev, hot_dev, test_softs)
test = data_utils.TensorDataset(word_pad_dev_tens, word_iis_dev_tens, char_pad_dev_tens, hot_dev_tens, soft_dev_tens)
test_loader = data_utils.DataLoader(test, batch_size=batsize, shuffle=False)

"""**The Model**"""

class Word_Encoder(torch.nn.Module):
    def __init__(self, lstm_size, embedding_size):
        super().__init__()

        self.bilstm = torch.nn.LSTM(embedding_size, lstm_size, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, wword_pad, col_indices):
        embedded_words = lookup_embeddings(word_embedding_lookup, wword_pad)
        rnn_context, _ = self.bilstm(embedded_words)
        rnn_sequence = torch.stack([torch.index_select(seq, 0, i) for seq, i in zip(rnn_context, col_indices)], 0)
        rnn_sequence = self.dropout(rnn_sequence)
        return rnn_sequence, rnn_context


class Char_Encoder(torch.nn.Module):
    def __init__(self, lstm_size, embedding_size):
        super().__init__()

        self.bilstm = torch.nn.LSTM(embedding_size, lstm_size, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, cchar_pad):
        embedded_chars = lookup_embeddings(char_embedding_lookup, cchar_pad)
        rnn_sequence, _ = self.bilstm(embedded_chars)
        rnn_sequence = self.dropout(rnn_sequence[:,1])
        reshaped = torch.reshape(rnn_sequence, [-1, 1, rnn_sequence.shape[1]])
        return reshaped


class Attention(torch.nn.Module):
    def __init__(self, attn_emb_dim, attn_size):
        super().__init__()

        self.attn_nn = torch.nn.Sequential(
                    torch.nn.Linear(attn_emb_dim, attn_size),
                    torch.nn.Tanh()
        )

        self.u_omega = torch.nn.Parameter(torch.randn([attn_size]))

    def forward(self, attn_in, s):
        v = self.attn_nn(attn_in)
        vu = torch.matmul(v.squeeze(1), self.u_omega)
        alphas = torch.nn.functional.softmax(vu, 0)
        final = torch.sum(attn_in * alphas.unsqueeze(-1), 1)
        return final



class RNN_all(torch.nn.Module):
    def __init__(self, exp='stl', smythe=None):
        super().__init__()

        self.word_encoder = Word_Encoder(lstm_size, word_emb_size)
        self.char_encoder = Char_Encoder(lstm_size, char_emb_size)

        self.char_attention = Attention(lstm_size*2, attn_size)
        self.word_attention = Attention(lstm_size*2, attn_size)

        concat_size = lstm_size*4
        hidden1 = int(lstm_size*4*sizeout_rate)
        out_final = int(hidden1*sizeout_rate)
        self.fulcon = torch.nn.Sequential(
                    torch.nn.Linear(concat_size, hidden1),
                    torch.nn.Linear(hidden1, out_final))

        self.output_hot = torch.nn.Linear(out_final, hotsize)


    def forward(self, wword_pad, wword_iis, cchar_pad, one_hot_labels, soft_labels, eval=True):
        word_sequence, word_context = self.word_encoder(wword_pad, wword_iis)
        word_attn= self.word_attention(word_sequence, 'word')

        char_sequence = self.char_encoder(cchar_pad)
        char_attn = self.char_attention(char_sequence, 'char')

        concat_attn = torch.cat([word_attn, char_attn], 1)
        ful = self.fulcon(concat_attn)

        pred_hot = self.output_hot(ful)  
        soft_labels = soft_labels + 1e-3
        softmax_scores = torch.nn.functional.softmax(pred_hot, 1) + 1e-43
        loss = torch.nn.MSELoss(reduction='sum')(soft_labels, softmax_scores)
        return softmax_scores, None, loss, 0.0

word_pad_trn_tens, word_iis_trn_tens, char_pad_trn_ten, hot_trn_tens, soft_trn_tens = create_dataset(word_pad_trn, word_iis_trn, char_pad_trn, hot_trn, train_srs)
train = data_utils.TensorDataset(word_pad_trn_tens, word_iis_trn_tens, char_pad_trn_ten, hot_trn_tens, soft_trn_tens)
train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=True)

print('Beginning the Training')
NUM_EXPERIMENTS = 30

accs = []
prfs = []
ct_prfs = []
jsds = []
kls = []
similarity_ents = []
ents_correlation = []
ce_results = []

dev_accs = []
dev_prfs = []

ct_dictionary = {}

for exp in range(NUM_EXPERIMENTS):
    print('\nExperiment %d #######################'%exp)
    best_val_f, best_val_acc = 0, 0
    best_val_r, best_val_p = 0, 0

    last_batch = 0

    model = RNN_all().cuda()
    model = to_cuda(model)
    optimizer = torch.optim.Adam(params=[p for p in model.parameters()],lr=0.001)

    for epoch in range(num_epochs):
        nepoch = epoch + 1
        model.train()

        for word_pad_trn_bat, word_iis_trn_bat, char_pad_trn_bat, y_hot_trn_bat, y_soft_trn_bat in train_loader:
            hard_predictions, _, hard_loss, _ = model(word_pad_trn_bat, word_iis_trn_bat, char_pad_trn_bat, y_hot_trn_bat, y_soft_trn_bat, False)
            backprop_hot(optimizer, hard_loss)
        
        # evaluate after each epoch using 
        dev_hard_preds, dev_soft_preds = get_predictions(model, dev_loader, False)
        dev_acc, dev_p, dev_r, dev_f = get_acc_f1(dev_hard_preds, np.argmax(hot_tst, 1))
        if dev_f > best_val_f:
            best_val_f = dev_f
            best_val_acc = dev_acc
            best_val_r = dev_r
            best_val_p = dev_p
            torch.save(model.state_dict(), DATA_PATH+'/best_model.pt')
        print(f"[Epoch {nepoch}] accuracy on dev: {dev_acc * 100:0.5f}, f1 on dev: {dev_f * 100:0.5f}")

    
    dev_accs.append(best_val_acc)
    dev_prfs.append([best_val_p, best_val_r, best_val_f])

    # evaluating on the test data
    del model
    model = RNN_all().cuda()
    model.load_state_dict(torch.load(DATA_PATH+'/best_model.pt'))
    test_predictions, test_preds_soft = get_predictions(model, test_loader, False)
    test_labels = np.argmax(hot_dev,1)

    ct_dictionary[str(exp)] = [item.tolist() for item in test_preds_soft]

    test_acc, test_p, test_r, test_f = get_acc_f1(test_labels, test_predictions)
    cp, cr, cf = get_ct_f1(test_labels, test_predictions, test_softs)

    jsd, kl = get_jsd_kl_div(test_softs, test_preds_soft)

    preds_ents = [entropy(p)/entropy(norm) for p in test_preds_soft]

    ent = cosine_similarity(np.array(test_entropys).reshape(1, NUM_TEST), np.array(preds_ents).reshape(1, NUM_TEST))[0][0]

    corr = np.corrcoef(test_entropys, preds_ents)[0][1]
    ce_res = cross_entropy(test_preds_soft, test_softs)
    
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

name = 'ct'
print('Training using ' + name)
import json
writepath = 'drive/My Drive/Colab Notebooks/Significance_Testing/pos_experiments/'

with open(writepath+'gimpelpos_' + name + '.jsonlines', 'w') as f:
    json.dump(ct_dictionary, f)

print('CT_SRS Accuracy stats after 30 experiments: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

print('\nCT_SRS PRF stats after 30 experiments:... ')
avgs = ['averages'] + np.average(prfs, 0).tolist()
maxs = ['maximums'] + np.max(prfs, 0).tolist()
mins = ['minimums'] + np.min(prfs, 0).tolist()
stds = ['stds'] + np.std(prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

print('\nCT_SRS Crowdtruth PRF stats after 30 experiments:... ')
avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
stds = ['stds'] + np.std(ct_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

print('\nCT_SRS JSD stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

print('\nCT_SRS KL stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

print('\nCT_SRS entropy similarity stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

print('\nCT_SRS entropy correlation stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ents_correlation), np.max(ents_correlation), np.min(ents_correlation), np.std(ents_correlation)))

print('\nCT_SRS crossentropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))

print('\n\nEVALUATING ON THE DEV SET I.E. THE FORNACIARA TEST SET')
print('Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(dev_accs)*100, np.max(dev_accs)*100, np.min(dev_accs)*100, np.std(dev_accs)*100))

print('\nPRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(dev_prfs, 0).tolist()
maxs = ['maximums'] + np.max(dev_prfs, 0).tolist()
mins = ['minimums'] + np.min(dev_prfs, 0).tolist()
stds = ['stds'] + np.std(dev_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))



print('CT_SRS Accuracy stats after 30 experiments: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

print('\nCT_SRS PRF stats after 30 experiments:... ')
avgs = ['averages'] + np.average(prfs, 0).tolist()
maxs = ['maximums'] + np.max(prfs, 0).tolist()
mins = ['minimums'] + np.min(prfs, 0).tolist()
stds = ['stds'] + np.std(prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

print('\nCT_SRS Crowdtruth PRF stats after 30 experiments:... ')
avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
stds = ['stds'] + np.std(ct_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

print('\nCT_SRS JSD stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

print('\nCT_SRS KL stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

print('\nCT_SRS entropy similarity stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

print('\nCT_SRS entropy correlation stats after 30 experiments: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ents_correlation), np.max(ents_correlation), np.min(ents_correlation), np.std(ents_correlation)))

print('\nCT_SRS crossentropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))

print('\n\nEVALUATING ON THE DEV SET I.E. THE FORNACIARA TEST SET')
print('Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(np.average(dev_accs)*100, np.max(dev_accs)*100, np.min(dev_accs)*100, np.std(dev_accs)*100))

print('\nPRF stats after 30 epochs:... ')
avgs = ['averages'] + np.average(dev_prfs, 0).tolist()
maxs = ['maximums'] + np.max(dev_prfs, 0).tolist()
mins = ['minimums'] + np.min(dev_prfs, 0).tolist()
stds = ['stds'] + np.std(dev_prfs, 0).tolist()
print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

