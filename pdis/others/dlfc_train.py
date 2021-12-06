import random
from collections import defaultdict
import time
import numpy as np
from dlfc_coref import *
import torch.optim as optim
import utils
import model_parameters
import json
import h5py
from colab_eval_metrics import *
from scipy.special import softmax


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer(object):
    """Trains and evaluates the model"""
    def __init__(self, train_path, dev_path, test_path, lr=0.001):
        self.__dict__.update(locals())
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.filter_widths = model_parameters.filter_widths
        self.char_dict = utils.load_char_dict(model_parameters.char_vocab_path)
        self.context_embeddings = utils.EmbeddingDictionary(model_parameters.glove_300d_filtered_pd)
        self.head_embeddings = utils.EmbeddingDictionary(model_parameters.glove_300d_2w_filtered_pd, maybe_cache=self.context_embeddings)
        self.class_names = model_parameters.class_names
        self.num_classes = len(self.class_names)
        self.lm_size = 1024
        self.lm_layers = 3
        self.lm_file = h5py.File(model_parameters.lm_path, "r")
        self.num_users = model_parameters.num_users
        self.genres = model_parameters.genres
        self.max_word_length = model_parameters.max_word_length


    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        if doc_key in ['wi/19033', 'wi/cdscs10b']:
            doc_key = 'gu/' + doc_key.split('/')[1]
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb


    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)


    def tensorize_document(self, example, is_training, gold_eval=False):
        hou_gold_mentions = example['mentions']
        sentences = example["sentences"]

        num_words = sum(len(s) for s in sentences)

        max_sentence_length = max(len(s) for s in sentences)

        # I fixed maximum word lenghth to make my CNN easier to handle. Todo: look into having variable max_word_lenght, see CharCNN in coref for more details.
        max_word_length = self.max_word_length

        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
           for j, word in enumerate(sentence):
               tokens[i][j] = word
               context_word_emb[i, j] = self.context_embeddings[word]
               head_word_emb[i, j] = self.head_embeddings[word]
               char_index[i, j, :min(max_word_length,len(word))] = [self.char_dict[c] for c in word[:max_word_length]]
        tokens = np.array(tokens)

        if is_training:
            annotations = example['annotations']
            max_annotation_length = max(len(a) for a in annotations)
            anno_len = np.array([len(a) for a in annotations])
            anno_user = np.zeros([len(anno_len), max_annotation_length]) -1
            anno_status = np.zeros([len(anno_len),max_annotation_length]) -1
            for i, m_anno in enumerate(annotations):
                for j, anno in enumerate(m_anno):
                    user_id, status, _,_ = anno
                    anno_user[i,j] = user_id
                    anno_status[i,j] = status
            mv_distrs = np.zeros_like(anno_status)
        else:
            num_classes = len(self.class_names)
            mv_distrs = np.array(example['mv_labels'])
            soft_labels = np.array(example['mace_posterior']) + EPSILON
          
            if gold_eval: # for testing on gold
                targets = np.argmax(np.array(example['gold_labels']), 1)
            else:
                targets = np.argmax(np.array(example['mace_posterior']), 1)
            labels = np.eye(num_classes)[targets]

        external_weights = np.random.randn(3,4) # just a dummy, it isn't used in dlfc        
        features = np.array(example['features'])

        doc_key = example["doc_key"]
        lm_emb = self.load_lm_embeddings(doc_key)

        genre = doc_key[3:]    # not using genre as a feature so I'm storing doc_name in the genre variable
        gold_starts, gold_ends = self.tensorize_mentions(hou_gold_mentions)

        if is_training:
            example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, genre, is_training, gold_starts, gold_ends, anno_user, features, external_weights, anno_status,mv_distrs)
        else:
            example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, genre, is_training, gold_starts, gold_ends, labels, features, external_weights, soft_labels,mv_distrs)
        return example_tensors


    def time_used(self, start_time):
        curr_time = time.time()
        used_time = curr_time-start_time
        m = used_time // 60
        s = used_time - 60 * m
        return "%d m %d s" % (m, s)


    def get_data(self, filepath):
        with open(filepath) as f:
            data = [json.loads(jsonline) for jsonline in f.readlines()]
        return data

    def create_dlfc_dataset(self, e):
        tokens, context_word_emb, head_word_emb  = e[0], torch.from_numpy(e[1]).float().to(device), torch.from_numpy(e[2]).float().to(device)
        lm_emb, char_index, text_len = torch.from_numpy(e[3]).float().to(device), torch.from_numpy(e[4]).long().to(device), torch.from_numpy(e[5]).int().to(device)
        genre, is_training, gold_starts, gold_ends = e[6], e[7], torch.from_numpy(e[8]).long().to(device), torch.from_numpy(e[9]).long().to(device)
        anno_users, features, external_weights = torch.from_numpy(e[10]).long().to(device), torch.from_numpy(e[11]).float().to(device), torch.from_numpy(e[12]).float().to(device)
        anno_status = torch.from_numpy(e[13]).long().to(device)
        mv_distrs = torch.from_numpy(e[14]).float().to(device)
        return tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, genre, is_training, gold_starts, gold_ends, anno_users, features, external_weights, anno_status, mv_distrs


    def create_dataset(self, e):
        tokens, context_word_emb, head_word_emb  = e[0], torch.from_numpy(e[1]).float().to(device), torch.from_numpy(e[2]).float().to(device)
        lm_emb, char_index, text_len = torch.from_numpy(e[3]).float().to(device), torch.from_numpy(e[4]).long().to(device), torch.from_numpy(e[5]).int().to(device)
        genre, is_training, gold_starts, gold_ends = e[6], e[7], torch.from_numpy(e[8]).long().to(device), torch.from_numpy(e[9]).long().to(device)
        mention_labels, features, external_weights = torch.from_numpy(e[10]).float().to(device), torch.from_numpy(e[11]).float().to(device), torch.from_numpy(e[12]).float().to(device)
        soft_labels = torch.from_numpy(e[13]).float().to(device)
        mv_distrs = torch.from_numpy(e[14]).float().to(device)
        return tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, genre, is_training, gold_starts, gold_ends, mention_labels, features, external_weights, soft_labels, mv_distrs


    def backprop(self, optimizer, hard_loss, soft_loss, exp):
        optimizer.zero_grad()
        if exp=='mtl':
            soft_loss.backward(retain_graph=True)
        hard_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        return


    def run_experiment(self, exp, smythe, train_documents, dev_documents, test_documents, num_epochs, num_experiments=10):
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

        prediction_dictionary = {}

        for exp_idx in range(num_experiments):
            print('\nExperiment %d #######################'%exp_idx)
            best_val_f, best_val_acc = 0, 0
            best_val_r, best_val_p = 0, 0

            last_batch = 0

            model = ISModel(exp, smythe)
            model = to_cuda(model)
            optimizer = optim.Adam(params=[p for p in model.parameters()
                                           if p.requires_grad],
                                   lr=0.001)

            for epoch in range(num_epochs):
                nepoch = epoch + 1
                model.train()

                random.shuffle(train_documents)
                for tensorized_document in train_documents:
                    tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, genre, is_training, gold_starts, gold_ends, \
                                                                    anno_users, features, external_weights, anno_status, mv_distrs= self.create_dlfc_dataset(tensorized_document)
                    _, _, l1, l2 = model(context_word_emb, head_word_emb, lm_emb, char_index, text_len, gold_starts, gold_ends, features, anno_users, anno_status, is_training)
                    self.backprop(optimizer, l1, l2, exp)


                # evaluate after each epoch using
                dev_hard, _, dev_pred_hard, _,_ = self.get_predictions(model, dev_documents, exp=='mtl')
                dev_acc, dev_p, dev_r, dev_f = get_acc_f1(dev_hard, dev_pred_hard, num_classes=len(self.class_names))

                if dev_f > best_val_f:
                    best_val_f = dev_f
                    best_val_acc = dev_acc
                    best_val_r = dev_r
                    best_val_p = dev_p
                    torch.save(model.state_dict(), 'best_model.pt')
                print(f"[Epoch {nepoch}] accuracy on dev: {dev_acc * 100:0.5f}, f1 on dev: {dev_f * 100:0.5f}")

                dev_accs.append(best_val_acc)
                dev_prfs.append([best_val_p, best_val_r, best_val_f])

            # evaluating on the test data
            del model
            model = to_cuda(ISModel(exp, smythe))
            model.load_state_dict(torch.load('best_model.pt'))
            test_hard, test_softs, test_pred_hard, test_preds_soft, test_distrs = self.get_predictions(model, test_documents, exp=='mtl')

            prediction_dictionary[str(exp_idx)] = [item.tolist() for item in test_preds_soft]

            test_acc, test_p, test_r, test_f = get_acc_f1(test_hard, test_pred_hard)
            cp, cr, cf = get_ct_f1(test_hard, test_pred_hard, test_distrs)

            jsd, kl = get_jsd_kl_div(test_softs, test_preds_soft)

            norm = [1/self.num_classes for i in range(self.num_classes)]
            preds_ents = [entropy(p)/entropy(norm) for p in test_preds_soft]
            test_entropys = [entropy(p)/entropy(norm) for p in test_softs]
            NUM_TEST = len(test_entropys)
            print(NUM_TEST)
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
           
        exp_name = 'DLFC'

        name = 'dlc'
        writepath = './predictions/'
        with open(writepath+'pdis_' + name+'.jsonlines', 'w') as f:
            json.dump(prediction_dictionary, f)

        print('\n%s Accuracy stats after 30 epochs: Avg %0.2f, Max %0.2f, Min %0.2f, Std %0.2f' %(exp_name, np.average(accs)*100, np.max(accs)*100, np.min(accs)*100, np.std(accs)*100))

        print('\n%s PRF stats after 30 epochs:... '%exp_name)
        avgs = ['averages'] + np.average(prfs, 0).tolist()
        maxs = ['maximums'] + np.max(prfs, 0).tolist()
        mins = ['minimums'] + np.min(prfs, 0).tolist()
        stds = ['stds'] + np.std(prfs, 0).tolist()
        print(tabulate([avgs, maxs, mins, stds], headers=['Precision', 'Recall', 'F1']))

        print('\n%s Crowdtruth PRF stats after 30 epochs:... '%exp_name)
        avgs = ['averages'] + np.average(ct_prfs, 0).tolist()
        maxs = ['maximums'] + np.max(ct_prfs, 0).tolist()
        mins = ['minimums'] + np.min(ct_prfs, 0).tolist()
        stds = ['stds'] + np.std(ct_prfs, 0).tolist()
        print(tabulate([avgs, maxs, mins, stds], headers=['CT Precision', 'CT Recall', 'CT F1']))

        print('\n%s JSD stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(exp_name, np.average(jsds), np.max(jsds), np.min(jsds), np.std(jsds)))

        print('\n%s KL stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(exp_name, np.average(kls), np.max(kls), np.min(kls), np.std(kls)))

        print('\n%s entropy similarity stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(exp_name, np.average(similarity_ents), np.max(similarity_ents), np.min(similarity_ents), np.std(similarity_ents)))

        print('\n%s entropy correlation stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(exp_name, np.average(ents_correlation), np.max(ents_correlation), np.min(ents_correlation), np.std(ents_correlation)))
        print('\nDLFC crossentropy stats after 30 epochs: Avg %0.4f, Max %0.4f, Min %0.4f, Std %0.4f' %(np.average(ce_results), np.max(ce_results), np.min(ce_results), np.std(ce_results)))
    
        print('\n\n')
        return


    def train(self, num_epochs=10, num_experiments=10):
        print('Tensorizing all documents')
        train_documents = [self.tensorize_document(document, True) for document in self.get_data(self.train_path)]
        dev_documents = [self.tensorize_document(document, False) for document in self.get_data(self.dev_path)]
        test_documents = [self.tensorize_document(document, False, True) for document in self.get_data(self.test_path)]
        print('Done!')

        for exp, smythe in [('stl', None)]:
            self.run_experiment(exp, smythe, train_documents, dev_documents, test_documents, num_epochs, num_experiments)
        return


    def get_predictions(self, model, eval_documents, mtl=False):
        hard_preds = []
        soft_preds = []
        distrs = []
        hards = []
        softs = []
        model.eval()
        for tensorized_document in eval_documents:
            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, genre, is_training, gold_starts, gold_ends, one_hot_labels, features, external_weights, soft_labels, mv_distrs = self.create_dataset(tensorized_document)
            if mtl:
                one_hot_pred, soft_pred, _, _ = model(context_word_emb, head_word_emb, lm_emb, char_index, text_len, gold_starts, gold_ends, features, one_hot_labels, soft_labels, is_training)
                hard_preds.extend(one_hot_pred.argmax(-1).detach().cpu().numpy())
                preds_soft = soft_pred.detach().cpu().numpy()
                preds_soft = np.zeros_like(preds_soft, dtype='float64') + preds_soft # for the JSD evaluation
                soft_preds.extend(preds_soft)
            else:
                one_hot_pred, _, _, _ = model(context_word_emb, head_word_emb, lm_emb, char_index, text_len, gold_starts, gold_ends, features, one_hot_labels, soft_labels, is_training)
                hard_preds.extend(one_hot_pred.argmax(-1).detach().cpu().numpy())
                preds_soft = one_hot_pred.detach().cpu().numpy()
                preds_soft = np.zeros_like(preds_soft, dtype='float64') + preds_soft
                soft_preds.extend(preds_soft)
            hards.extend(one_hot_labels.argmax(-1).cpu().numpy())
            softs.extend(soft_labels.cpu().numpy())
            distrs.extend(mv_distrs.cpu().numpy())
        return hards, softs, hard_preds, soft_preds, distrs

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model_parameters.train_path, model_parameters.dev_path, model_parameters.test_path)
    trainer.train(10, 10)

