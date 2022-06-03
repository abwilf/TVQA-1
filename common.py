import torch
from torch import nn
import math

def get_fc_edges(edges_a, edges_b):
    return torch.cat([elt[None,:] for elt in torch.meshgrid(edges_a, edges_b)]).reshape(2,-1)

def get_fc_edges_window(idx1, idx2, window):
    # for "pres" nodes,only get indices where they are similar in their index relative to their seq len: e.g. 
    # seq1 = 0...9 -> 0/9,1/9...etc
    # seq2 = 0...20 -> 0/20,1/20...etc
    # window size then corresponds to 3 (#window) of this sequence to the left or right.  e.g. in the shorter sequence, it would look for +- 3/9 and see which ones in seq2 correspond
    # Note: this is no longer symmetrical, which it shouldn't be b/c that doesn't make sense with unaligned seqs

    # set idx1 and idx2 to 0, then turn into relatives
    idx1_start = idx1.min()
    idx2_start = idx2.min()
    idx1 = idx1 - idx1_start
    idx2 = idx2 -idx2_start
    idx1 = idx1 / len(idx1) # turn into relatives
    idx2 = idx2 / len(idx2)

    if window > len(idx1):
        window = len(idx1)
    
    window = (1/len(idx1))*window # e.g. consider all seq2 within +- 3/9

    arr = torch.cat([elt[None,:] for elt in torch.meshgrid(idx1, idx2)]).reshape(2,-1)
    valid_idxs = torch.where(torch.abs(arr[1,:]-arr[0,:]) <= window)[0]
    arr = arr[:,valid_idxs]

    arr[0,:] = arr[0,:]*len(idx1) + idx1_start
    arr[1,:] = arr[1,:]*len(idx2) + idx2_start
    arr = torch.Tensor(np.round(arr,decimals=0)).to(torch.long)
    return arr


    
def get_fc_edges_fut_window(idx1, idx2, window):
    # for "pres" nodes,only get indices where they are similar in their index relative to their seq len: e.g. 
    # seq1 = 0...9 -> 0/9,1/9...etc
    # seq2 = 0...20 -> 0/20,1/20...etc
    # window size then corresponds to 3 (#window) of this sequence to the left or right.  e.g. in the shorter sequence, it would look for +- 3/9 and see which ones in seq2 correspond
    # Note: this is no longer symmetrical, which it shouldn't be b/c that doesn't make sense with unaligned seqs

    # set idx1 and idx2 to 0, then turn into relatives
    try:
        idx1_start = idx1.min()
        idx2_start = idx2.min()
    except:
        hi=2
    idx1 = idx1 - idx1_start
    idx2 = idx2 -idx2_start
    idx1 = idx1 / len(idx1) # turn into relatives
    idx2 = idx2 / len(idx2)

    if window > len(idx1):
        window = len(idx1)
    
    window = (1/len(idx1))*window

    arr = torch.cat([elt[None,:] for elt in torch.meshgrid(idx1, idx2)]).reshape(2,-1)
    
    valid_idxs = torch.where( ( (arr[1,:]-arr[0,:]) >= window ) & ( (arr[1,:]-arr[0,:]) <= 2*window ))[0] # this is the key difference w/above
    arr = arr[:,valid_idxs]

    arr[0,:] = arr[0,:]*len(idx1) + idx1_start
    arr[1,:] = arr[1,:]*len(idx2) + idx2_start
    arr = torch.Tensor(np.round(arr,decimals=0)).to(torch.long)

    return arr

def get_fc_edges_past_window(idx1, idx2, window):
    # for "pres" nodes,only get indices where they are similar in their index relative to their seq len: e.g. 
    # seq1 = 0...9 -> 0/9,1/9...etc
    # seq2 = 0...20 -> 0/20,1/20...etc
    # window size then corresponds to 3 (#window) of this sequence to the left or right.  e.g. in the shorter sequence, it would look for +- 3/9 and see which ones in seq2 correspond
    # Note: this is no longer symmetrical, which it shouldn't be b/c that doesn't make sense with unaligned seqs

    # set idx1 and idx2 to 0, then turn into relatives
    idx1_start = idx1.min()
    idx2_start = idx2.min()
    idx1 = idx1 - idx1_start
    idx2 = idx2 -idx2_start
    idx1 = idx1 / len(idx1) # turn into relatives
    idx2 = idx2 / len(idx2)

    if window > len(idx1):
        window = len(idx1)
    
    window = (1/len(idx1))*window

    arr = torch.cat([elt[None,:] for elt in torch.meshgrid(idx1, idx2)]).reshape(2,-1)
    
    valid_idxs = torch.where( ( (arr[1,:]-arr[0,:]) <= -1*window ) & ( (arr[1,:]-arr[0,:]) >= -2*window ))[0] # this is the key difference w/above
    arr = arr[:,valid_idxs]

    arr[0,:] = arr[0,:]*len(idx1) + idx1_start
    arr[1,:] = arr[1,:]*len(idx2) + idx2_start
    arr = torch.Tensor(np.round(arr,decimals=0)).to(torch.long)

    return arr


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Added to support odd d_model
        pe = pe.unsqueeze(0).transpose(0, 1).squeeze()
        self.register_buffer('pe', pe)

    def forward(self, x, counts):
        # counts is the count for each element in the batch, x is of shape
        pe_rel = torch.cat([self.pe[:count,:] for count in counts])
        x = x + pe_rel.to('cuda')
        return self.dropout(x)
        
