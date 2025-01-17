__author__ = "Jie Lei"

import torch
from torch import nn

from .rnn import RNNEncoder, max_along_time
from .bidaf import BidafAttn
from .mlp import MLP

import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
import torch_geometric
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import torch_geometric.transforms as T
from common import *
from torch_geometric.nn.conv import HeteroConv
from sklearn.metrics import accuracy_score


class Solograph_HeteroGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        assert gc['hidden_dim'] % gc['num_heads'] == 0, 'Hidden channels must be divisible by number of heads'


        self.convs = torch.nn.ModuleList()
        if gc['batch_norm']:
            self.batchnorms = torch.nn.ModuleList()
        
        for i in range(gc['num_convs']):
            sub_a_conv = GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het'])
            a_sub_conv = GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het'])
            a_a_conv = GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het'])
            q_a_conv = GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het'])
            a_q_conv = GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het'])
            
            conv_dict = {
                # self conns
                ('q', 'q_q', 'q'): GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het']),
                ('sub', 'sub_sub', 'sub'): GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het']),
                ('a0', 'a_a', 'a0'): a_a_conv,
                ('a1', 'a_a', 'a1'): a_a_conv,
                ('a2', 'a_a', 'a2'): a_a_conv,
                ('a3', 'a_a', 'a3'): a_a_conv,
                ('a4', 'a_a', 'a4'): a_a_conv,

                # sub
                ('sub', 'sub_q', 'q'): GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het']),
                ('q', 'q_sub', 'sub'): GATv2Conv(gc['hidden_dim'], gc['hidden_dim']//gc['num_heads'], heads=gc['num_heads'], dropout=gc['drop_het']),
                
                ('sub', 'sub_a', 'a0'): sub_a_conv,
                ('sub', 'sub_a', 'a1'): sub_a_conv,
                ('sub', 'sub_a', 'a2'): sub_a_conv,
                ('sub', 'sub_a', 'a3'): sub_a_conv,
                ('sub', 'sub_a', 'a4'): sub_a_conv,

                ('a0', 'a_sub', 'sub'): a_sub_conv,
                ('a1', 'a_sub', 'sub'): a_sub_conv,
                ('a2', 'a_sub', 'sub'): a_sub_conv,
                ('a3', 'a_sub', 'sub'): a_sub_conv,
                ('a4', 'a_sub', 'sub'): a_sub_conv,

                # qa
                ('q', 'q_a', 'a0'): q_a_conv,
                ('q', 'q_a', 'a1'): q_a_conv,
                ('q', 'q_a', 'a2'): q_a_conv,
                ('q', 'q_a', 'a3'): q_a_conv,
                ('q', 'q_a', 'a4'): q_a_conv,

                ('a0', 'a_q','q'): a_q_conv,
                ('a1', 'a_q','q'): a_q_conv,
                ('a2', 'a_q','q'): a_q_conv,
                ('a3', 'a_q','q'): a_q_conv,
                ('a4', 'a_q','q'): a_q_conv,
            }

            conv = HeteroConv(conv_dict, aggr='mean')

            self.convs.append(conv)
            if gc['batch_norm']:
                bn = torch_geometric.nn.norm.BatchNorm(gc['hidden_dim'])
                self.batchnorms.append(bn)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for i,conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
            if gc['batch_norm']:
                x_dict = {k: self.batchnorms[i](v) for k,v in x_dict.items()}

        return x_dict

class ABC(nn.Module):
    def __init__(self, _gc):
        super(ABC, self).__init__()
        global gc; gc=_gc
        self.vid_flag = "imagenet" in gc['input_streams']
        self.sub_flag = "sub" in gc['input_streams']
        self.vcpt_flag = "vcpt" in gc['input_streams']
        hidden_size_1 = gc['hidden_dim']
        hidden_size_2 = gc['hidden_dim']
        n_layers_cls = gc['n_layers_cls']
        vid_feat_size = gc['vid_feat_size']
        embedding_size = gc['embedding_size']
        vocab_size = gc['vocab_size']

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size_1 * 3, method="dot")  # no parameter for dot
        self.lstm_raw = RNNEncoder(300, hidden_size_1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if self.vid_flag:
            print("activate video stream")
            self.video_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(vid_feat_size, embedding_size),
                nn.Tanh(),
            )
            self.lstm_mature_vid = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vid = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        if self.sub_flag:
            print("activate sub stream")
            self.lstm_mature_sub = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_sub = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.lstm_mature_vcpt = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                               dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vcpt = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        def get_lin():
            return nn.Sequential(
                nn.Linear(300, gc['hidden_dim']),
                nn.Dropout(.1),
                nn.ReLU(),
            )

        self.lin_dict = torch.nn.ModuleDict()
        self.lin_dict['q'] = get_lin()
        self.lin_dict['a'] = get_lin()
        self.lin_dict['sub'] = get_lin()

        self.hetero_gnn = Solograph_HeteroGNN()
        self.pe = PositionalEncoding(gc['hidden_dim'])

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l,
                sub, sub_l, vcpt, vcpt_l, vid, vid_l, batch):
        # NOTE: this is a late fusion model.  It processes each stream (e.g. video, visual concept region, subtitle) independently, then adds their predictions together.
        # q, a are token of shape [bs, seq_len] = [100, 25] where each elt is an integer (token number)
        # *_l is the length of each question / answer.  This is a tensor of shape [100,] of dtype integer > 0

        # e_* is an embedding vector: using GLoVE to turn token numbers into word vectors of shape [bs,seq_len,300]
        e_q = q
        e_a0 = a0
        e_a1 = a1
        e_a2 = a2
        e_a3 = a3
        e_a4 = a4

        # e_q = self.embedding(q)
        # e_a0 = self.embedding(a0)
        # e_a1 = self.embedding(a1)
        # e_a2 = self.embedding(a2)
        # e_a3 = self.embedding(a3)
        # e_a4 = self.embedding(a4)

        # contextualized q/a versions from bi-lstm. same shape: [bs,seq_len,300]
        # raw_out_q, _ = self.lstm_raw(e_q, q_l)
        # raw_out_a0, _ = self.lstm_raw(e_a0, a0_l)
        # raw_out_a1, _ = self.lstm_raw(e_a1, a1_l)
        # raw_out_a2, _ = self.lstm_raw(e_a2, a2_l)
        # raw_out_a3, _ = self.lstm_raw(e_a3, a3_l)
        # raw_out_a4, _ = self.lstm_raw(e_a4, a4_l)

        if self.sub_flag:
            # run lin projection
            batch['sub']['x'] = self.lin_dict['sub'](batch['sub']['x'])
            batch['q']['x'] = self.lin_dict['q'](batch['q']['x'])
            for a in ['a0', 'a1', 'a2', 'a3', 'a4']:
                batch[a]['x'] = self.lin_dict['a'](batch[a]['x'])

            # positional encoding
            for k in ['sub', 'q', 'a0', 'a1', 'a2', 'a3', 'a4']:
                counts = torch.diff(batch[k]['ptr'])
                batch[k]['x'] = self.pe(batch[k]['x'], counts)

            # run convs
            x_dict = self.hetero_gnn(batch.x_dict, batch.edge_index_dict, None)

            ## pooled rep
            # run linear clf
            # get all end of sentence tokens
            qa_keys = ['q', 'a0', 'a1', 'a2', 'a3', 'a4']
            clf_dict = {k: x_dict[k][batch[k]['ptr'][1:]-1] for k in qa_keys} # eos tokens

            a_keys = ['a0', 'a1', 'a2', 'a3', 'a4']
            mature_answers = torch.cat([torch.cat([clf_dict['q'], clf_dict[a]], -1)[:,None,:] for a in a_keys], 1)
            
            out = self.classifier_sub(mature_answers)  # (B, 5)
            return out.squeeze()
            
            ## try it with original setup - hard b/c adding padding might screw things up in backward pass?
            # sub_l = counts
            # sub_out = self.stream_processor(self.lstm_mature_sub, self.classifier_sub, sub_out, sub_l,
            #                                 raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
            #                                 raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)


            #

        else:
            sub_out = 0

        if self.vcpt_flag:
            e_vcpt = vcpt
            raw_out_vcpt, _ = self.lstm_raw(e_vcpt, vcpt_l)
            vcpt_out = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, raw_out_vcpt, vcpt_l,
                                             raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                             raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag:
            e_vid = self.video_fc(vid)
            try:
                raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            except:
                hi=2
            vid_out = self.stream_processor(self.lstm_mature_vid, self.classifier_vid, raw_out_vid, vid_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vid_out = 0

        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        return out.squeeze()

    def stream_processor(self, lstm_mature, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):
        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)

        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        mature_maxout_a0, _ = lstm_mature(concat_a0, ctx_l)
        mature_maxout_a1, _ = lstm_mature(concat_a1, ctx_l)
        mature_maxout_a2, _ = lstm_mature(concat_a2, ctx_l)
        mature_maxout_a3, _ = lstm_mature(concat_a3, ctx_l)
        mature_maxout_a4, _ = lstm_mature(concat_a4, ctx_l)

        mature_maxout_a0 = max_along_time(mature_maxout_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(mature_maxout_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(mature_maxout_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(mature_maxout_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(mature_maxout_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out

    @staticmethod
    def get_fake_inputs(device="cuda:0"):
        bs = 16
        q = torch.ones(bs, 25).long().to(device)
        q_l = torch.ones(bs).fill_(25).long().to(device)
        a = torch.ones(bs, 5, 20).long().to(device)
        a_l = torch.ones(bs, 5).fill_(20).long().to(device)
        a0, a1, a2, a3, a4 = [a[:, i, :] for i in range(5)]
        a0_l, a1_l, a2_l, a3_l, a4_l = [a_l[:, i] for i in range(5)]
        sub = torch.ones(bs, 300).long().to(device)
        sub_l = torch.ones(bs).fill_(300).long().to(device)
        vcpt = torch.ones(bs, 300).long().to(device)
        vcpt_l = torch.ones(bs).fill_(300).long().to(device)
        vid = torch.ones(bs, 100, 2048).to(device)
        vid_l = torch.ones(bs).fill_(100).long().to(device)
        return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l


if __name__ == '__main__':
    from config import BaseOptions
    import sys
    sys.argv[1:] = ["--input_streams" "sub"]
    opt = BaseOptions().parse()

    model = ABC(opt)
    model.to(gc['device'])
    test_in = model.get_fake_inputs(device=gc['device'])
    test_out = model(*test_in)
    print(test_out.size())
