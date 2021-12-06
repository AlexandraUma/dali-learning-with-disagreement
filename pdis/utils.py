import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import codecs
from collections import defaultdict


def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()


def to_numpy(torch_tensor):
    return torch_tensor.cpu().clone().detach().numpy()


def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def safe_divide(x, y):
    """ Make sure we don't divide by 0 """
    if y != 0:
        return x / y
    return 1


def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]


def mask_len(item, max_len):
    return (to_cuda(torch.arange(max_len).expand(len(item), max_len)) < item.unsqueeze(1))


def flatten_emb_by_sentence(text_embedding, text_lens):
    flattened_emb = []
    for i, sent_len in enumerate(text_lens):
        flattened_emb.append(text_embedding[i][:sent_len])
    return torch.cat(flattened_emb, 0)


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = defaultdict(int)
    char_dict.update({c:i for i, c in enumerate(vocab)})
    return char_dict



class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._normalize = normalize
        self._path = info["path"]
        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path)

     
    @property
    def size(self):
        return self._size


    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        embedding_dict = defaultdict(lambda:default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path) as f:
                for i, line in enumerate(f.readlines()):
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
                    assert len(embedding) == self.size
                    embedding_dict[word] = embedding
            if vocab_size is not None:
               assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict


    def __getitem__(self, key):
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding


    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v

