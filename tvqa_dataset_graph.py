__author__ = "Jie Lei"

import h5py
import numpy as np
import torch
from torch import nn
# from torch.utils.data.dataset import Dataset
from torch_geometric.data import Dataset

from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist

import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import rmrf,mkdirp

from torch_geometric.data import HeteroData
gc = None

from common import *

def tolong(x):
    return torch.Tensor(x).to(torch.long)

indices = []
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *

class TVQADataset(Dataset):
    def __init__(self, _gc, mode="train"):
        global gc; gc=_gc

        self.raw_train = load_json(gc['train_path'])
        self.raw_test = load_json(gc['test_path'])
        self.raw_valid = load_json(gc['valid_path'])
        self.vcpt_dict = load_pickle(gc['vcpt_path'])
        self.vfeat_load = gc['vid_feat_flag']
        if self.vfeat_load:
            self.vid_h5 = h5py.File(gc['vid_feat_path'], "r", driver=gc['h5driver'])
        self.glove_embedding_path = gc['glove_path']
        self.normalize_v = gc['normalize_v']
        self.with_ts = gc['with_ts']
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

        # set word embedding / vocabulary
        self.word2idx_path = gc['word2idx_path']
        self.idx2word_path = gc['idx2word_path']
        self.vocab_embedding_path = gc['vocab_embedding_path']
        self.embedding_dim = gc['embedding_size']
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.offset = len(self.word2idx)

        # set entry keys
        if self.with_ts:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
        else:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_text"]
        self.vcpt_key = "vcpt"
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"
        self.located_frm_key = "located_frame"
        for k in self.text_keys + [self.vcpt_key, self.qid_key, self.vid_name_key]:
            if k == "vcpt":
                continue
            assert k in self.raw_valid[0].keys()

        # build/load vocabulary
        if not files_exist([self.word2idx_path, self.idx2word_path, self.vocab_embedding_path]):
            print("\nNo cache founded.")
            self.build_word_vocabulary(word_count_threshold=gc['word_count_threshold'])
        else:
            print("\nLoading cache ...")
            self.word2idx = load_pickle(self.word2idx_path)
            self.idx2word = load_pickle(self.idx2word_path)
            self.vocab_embedding = load_pickle(self.vocab_embedding_path)
        
        self.embedding = nn.Embedding(len(self.word2idx), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(self.vocab_embedding))

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = []
        if self.with_ts:
            cur_start, cur_end = self.cur_data_dict[index][self.located_frm_key]
        cur_vid_name = self.cur_data_dict[index][self.vid_name_key]

        # add text keys
        for k in self.text_keys:
            items.append(self.numericalize(self.cur_data_dict[index][k]))

        # add vcpt
        if self.with_ts:
            cur_vis_sen = self.vcpt_dict[cur_vid_name][cur_start:cur_end + 1]
        else:
            cur_vis_sen = self.vcpt_dict[cur_vid_name]
        cur_vis_sen = " , ".join(cur_vis_sen)
        items.append(self.numericalize_vcpt(cur_vis_sen))

        # add other keys
        if self.mode == 'test':
            items.append(666)  # this value will not be used
        else:
            items.append(int(self.cur_data_dict[index][self.label_key]))
        for k in [self.qid_key]:
            items.append(self.cur_data_dict[index][k])
        items.append(cur_vid_name)

        # add visual feature
        if self.vfeat_load:
            if self.with_ts:
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][cur_start:cur_end])
            else:  # handled by vid_path
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][:480])
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)
        else:
            cur_vid_feat = torch.zeros([2, 2])  # dummy placeholder
        items.append(cur_vid_feat)

        item_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt", "label_key", "qid_key", "vid_name_key", "vid"]
        d = {k: {'x': v} for k,v in zip(item_keys,items)}
        
        label = d['label_key']['x']

        # filter non-tensor keys
        d = {k:v for k,v in d.items() if k not in ['qid_key', 'vid_name_key', 'label_key']}

        # format label key
        # d['label_key'] = tolong(d['label_key']['x'])

        graph_keys = ['sub', 'q', 'a0', 'a1', 'a2', 'a3', 'a4']

        if self.vfeat_load:
            # pad and clip vid feats
            k = 'vid'
            arr = d[k]['x']
            d[f'{k}_l'] = tolong([min(arr.shape[0], gc['max_vid_l'])])
            arr = arr[:gc['max_vid_l'],:]
            if k not in graph_keys: # pad
                arr = torch.nn.functional.pad(arr, (0,0,0,gc['max_vid_l']-arr.shape[0]))
            d[k] = {'x': arr}
        else:
            del d['vid']
        
        # get full text embeddings
        q_max = gc['max_q_l']
        text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
        max_pad = {
            "q": q_max, "a0": q_max, "a1": q_max, "a2": q_max, "a3": q_max, "a4": q_max,
            "sub": gc['max_sub_l'],
            "vcpt": gc['max_vcpt_l']
        }

        with torch.no_grad():
            for k in text_keys:
                arr = self.embedding(tolong(d[k]['x']))
                d[f'{k}_l'] = tolong([min(arr.shape[0], max_pad[k])])
                
                # pad and clip
                arr = arr[:max_pad[k],:]
                if k not in graph_keys: # pad
                    arr = torch.nn.functional.pad(arr,(0,0,0,max_pad[k]-arr.shape[0]))
                d[k]['x'] = arr

        idxs = {
            k: torch.arange(d[k]['x'].shape[0]) for k in graph_keys
        }
        d = {
            **d,
            
            # self conns
            ('q', 'q_q', 'q'): get_fc_edges(idxs['q'], idxs['q']),
            ('sub', 'sub_sub', 'sub'): get_fc_edges(idxs['sub'], idxs['sub']),
            ('a0', 'a_a', 'a0'): get_fc_edges(idxs['a0'], idxs['a0']),
            ('a1', 'a_a', 'a1'): get_fc_edges(idxs['a1'], idxs['a1']),
            ('a2', 'a_a', 'a2'): get_fc_edges(idxs['a2'], idxs['a2']),
            ('a3', 'a_a', 'a3'): get_fc_edges(idxs['a3'], idxs['a3']),
            ('a4', 'a_a', 'a4'): get_fc_edges(idxs['a4'], idxs['a4']),

            # sub
            ('sub', 'sub_q', 'q'): get_fc_edges(idxs['sub'], idxs['q']),
            ('q', 'q_sub', 'sub'): get_fc_edges(idxs['q'], idxs['sub']),
            
            ('sub', 'sub_a', 'a0'): get_fc_edges(idxs['sub'], idxs['a0']),
            ('sub', 'sub_a', 'a1'): get_fc_edges(idxs['sub'], idxs['a1']),
            ('sub', 'sub_a', 'a2'): get_fc_edges(idxs['sub'], idxs['a2']),
            ('sub', 'sub_a', 'a3'): get_fc_edges(idxs['sub'], idxs['a3']),
            ('sub', 'sub_a', 'a4'): get_fc_edges(idxs['sub'], idxs['a4']),

            ('a0', 'a_sub', 'sub'): get_fc_edges(idxs['a0'], idxs['sub']),
            ('a1', 'a_sub', 'sub'): get_fc_edges(idxs['a1'], idxs['sub']),
            ('a2', 'a_sub', 'sub'): get_fc_edges(idxs['a2'], idxs['sub']),
            ('a3', 'a_sub', 'sub'): get_fc_edges(idxs['a3'], idxs['sub']),
            ('a4', 'a_sub', 'sub'): get_fc_edges(idxs['a4'], idxs['sub']),

            # qa
            ('q', 'q_a', 'a0'): get_fc_edges(idxs['q'], idxs['a0']),
            ('q', 'q_a', 'a1'): get_fc_edges(idxs['q'], idxs['a1']),
            ('q', 'q_a', 'a2'): get_fc_edges(idxs['q'], idxs['a2']),
            ('q', 'q_a', 'a3'): get_fc_edges(idxs['q'], idxs['a3']),
            ('q', 'q_a', 'a4'): get_fc_edges(idxs['q'], idxs['a4']),

            ('a0', 'a_q','q'): get_fc_edges(idxs['a0'], idxs['q']),
            ('a1', 'a_q','q'): get_fc_edges(idxs['a1'], idxs['q']),
            ('a2', 'a_q','q'): get_fc_edges(idxs['a2'], idxs['q']),
            ('a3', 'a_q','q'): get_fc_edges(idxs['a3'], idxs['q']),
            ('a4', 'a_q','q'): get_fc_edges(idxs['a4'], idxs['q']),
        }

        d = {
            **d,
            **{k: {'edge_index': v} for k,v in d.items() if isinstance(k, tuple) },
        }

        d = HeteroData(d)
        return d, label

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them
        words = [w for w in words if w != ","]
        words = words + [eos_word] if eos else words
        return words

    def numericalize(self, sentence, eos=True):
        """convert words to indices"""
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
        return sentence_indices

    def numericalize_vcpt(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
        unique_pairs = []
        for pair in attr_obj_pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("<eos>")
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in words]
        return sentence_indices

    @classmethod
    def load_glove(cls, filename):
        """ Load glove embeddings into a python dict
        returns { word (str) : vector_embedding (torch.FloatTensor) }"""
        glove = {}
        with open(filename) as f:
            for line in f.readlines():
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def build_word_vocabulary(self, word_count_threshold=0):
        """borrowed this implementation from @karpathy's neuraltalk."""
        print("Building word vocabulary starts.\n")
        all_sentences = []
        for k in self.text_keys:
            all_sentences.extend([ele[k] for ele in self.raw_train])

        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.word2idx.keys()]
        print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
              (len(vocab), word_count_threshold))

        # build index and vocabularies
        for idx, w in enumerate(vocab):
            self.word2idx[w] = idx + self.offset
            self.idx2word[idx + self.offset] = w
        print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))

        # Make glove embedding.
        print("Loading glove embedding at path : %s. \n" % self.glove_embedding_path)
        glove_full = self.load_glove(self.glove_embedding_path)
        print("Glove Loaded, building word2idx, idx2word mapping. This may take a while.\n")
        glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
        glove_keys = glove_full.keys()
        for i in tqdm(range(len(self.idx2word))):
            w = self.idx2word[i]
            w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
            glove_matrix[i, :] = w_embed
        self.vocab_embedding = glove_matrix
        print("Vocab embedding size is :", glove_matrix.shape)

        print("Saving cache files ...\n")
        save_pickle(self.word2idx, self.word2idx_path)
        save_pickle(self.idx2word, self.idx2word_path)
        save_pickle(glove_matrix, self.vocab_embedding_path)
        print("Building  vocabulary done.\n")


class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        return batch


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    def pad_sequences(sequences):
        sequences = [torch.LongTensor(s) for s in sequences]
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq[:end]
        return padded_seqs, lengths

    def pad_video_sequences(sequences):
        """sequences is a list of torch float tensors (created from numpy)"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), v_dim).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        return padded_seqs, lengths

    # separate source and target sequences
    column_data = zip(*data)
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_name_key = "vid_name"
    vid_feat_key = "vid"
    all_keys = text_keys + [label_key, qid_key, vid_name_key, vid_feat_key]
    all_values = []
    column_data = list(column_data)
    for i, k in enumerate(all_keys):
        if k in text_keys:
            all_values.append(pad_sequences(column_data[i]))
        elif k == label_key:
            all_values.append(torch.LongTensor(column_data[i]))
        elif k == vid_feat_key:
            all_values.append(pad_video_sequences(column_data[i]))
        else:
            all_values.append(column_data[i])

    batched_data = Batch.get_batch(keys=all_keys, values=all_values)
    return batched_data


def preprocess_inputs(batched_data, max_sub_l, max_vcpt_l, max_vid_l, device="cuda:0"):
    """clip and move to target device"""
    max_len_dict = {"sub": max_sub_l, "vcpt": max_vcpt_l, "vid": max_vid_l}
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_feat_key = "vid"
    model_in_list = []
    for k in text_keys + [vid_feat_key]:
        v = getattr(batched_data, k)
        if k in max_len_dict:
            ctx, ctx_l = v
            max_l = min(ctx.size(1), max_len_dict[k])
            if ctx.size(1) > max_l:
                ctx_l = ctx_l.clamp(min=1, max=max_l)
                ctx = ctx[:, :max_l]
            model_in_list.extend([ctx.to(device), ctx_l.to(device)])
        else:
            model_in_list.extend([v[0].to(device), v[1].to(device)])
    target_data = getattr(batched_data, label_key)
    target_data = target_data.to(device)
    qid_data = getattr(batched_data, qid_key)
    return model_in_list, target_data, qid_data


if __name__ == "__main__":
    # python tvqa_dataset.py --input_streams sub
    import sys
    from config import BaseOptions
    sys.argv[1:] = ["--input_streams", "sub"]
    opt = BaseOptions().parse()

    rmrf('./cache')
    mkdirp('./cache')

    dset = TVQADataset(opt, mode="valid")
    data_loader = DataLoader(dset, batch_size=10, shuffle=False, collate_fn=pad_collate)

    for batch_idx, batch in enumerate(data_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, gc['max_sub_l'], gc['max_vcpt_l'], gc['max_vid_l'])
        break

