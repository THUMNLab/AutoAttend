'''
ops interface, borrowed from TextNAS
'''

import torch
import torch.nn.functional as F
from torch import nn

def get_length(mask):
    length = torch.sum(mask, 1)
    length = length.long()
    return length

INF = 1E10

class Mask(nn.Module):
    def forward(self, seq, mask):
        # seq: (N, C, L)
        # mask: (N, L)
        seq_mask = torch.unsqueeze(mask, 2)
        seq_mask = torch.transpose(seq_mask.repeat(1, 1, seq.size()[1]), 1, 2)
        return seq.where(torch.eq(seq_mask, 1), torch.zeros_like(seq))


class BatchNorm(nn.Module):
    def __init__(self, num_features, pre_mask, post_mask, eps=1e-5, decay=0.9, affine=True):
        super(BatchNorm, self).__init__()
        self.mask_opt = Mask()
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=1.0 - decay, affine=affine)

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.bn(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class ConvBN(nn.Module):
    def __init__(self, kernal_size, in_channels, out_channels, cnn_keep_prob,
                 pre_mask, post_mask, with_bn=True, with_relu=True, with_pre_norm=False):
        super(ConvBN, self).__init__()
        self.mask_opt = Mask()
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.with_pre_norm = with_pre_norm
        self.conv = nn.Conv1d(in_channels, out_channels, kernal_size, 1, bias=True, padding=(kernal_size - 1) // 2)
        self.dropout = nn.Dropout(p=(1 - cnn_keep_prob))

        if with_bn:
            self.bn = BatchNorm(out_channels, not post_mask, True)

        if with_relu:
            self.relu = nn.ReLU()
        
        if with_pre_norm:
            self.layerNorm = nn.LayerNorm(in_channels)

    def forward(self, seq, mask):
        if self.with_pre_norm:
            seq = self.layerNorm(seq.transpose(1,2)).transpose(1,2)
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.conv(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        if self.with_bn:
            seq = self.bn(seq, mask)
        if self.with_relu:
            seq = self.relu(seq)
        seq = self.dropout(seq)
        return seq


class AvgPool(nn.Module):
    def __init__(self, kernal_size, pre_mask, post_mask, with_pre_norm=False, dim=None):
        super(AvgPool, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernal_size, 1, padding=(kernal_size - 1) // 2)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.mask_opt = Mask()
        self.with_pre_norm = with_pre_norm
        if self.with_pre_norm:
            self.layerNorm = nn.LayerNorm(dim)

    def forward(self, seq, mask):
        if self.with_pre_norm:
            seq = self.layerNorm(seq.transpose(1,2)).transpose(1,2)
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.avg_pool(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class MaxPool(nn.Module):
    def __init__(self, kernal_size, pre_mask, post_mask, with_pre_norm=False, dim=None):
        super(MaxPool, self).__init__()
        self.max_pool = nn.MaxPool1d(kernal_size, 1, padding=(kernal_size - 1) // 2)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.mask_opt = Mask()
        self.with_pre_norm = with_pre_norm
        if self.with_pre_norm:
            self.layerNorm = nn.LayerNorm(dim)

    def forward(self, seq, mask):
        if self.with_pre_norm:
            seq = self.layerNorm(seq.transpose(1,2)).transpose(1,2)
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = seq.contiguous()
        seq = self.max_pool(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq

class Attention(nn.Module):
    def __init__(self, num_units, num_heads, keep_prob, is_mask, with_bn=True, with_pre_norm=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.keep_prob = keep_prob

        self.linear_q = nn.Linear(num_units, num_units, bias=False)
        self.linear_k = nn.Linear(num_units, num_units, bias=False)
        self.linear_v = nn.Linear(num_units, num_units, bias=False)

        self.o_net = nn.Linear(num_units, num_units, bias=False)

        self.with_bn = with_bn
        self.with_pre_norm = with_pre_norm
        if self.with_bn:
            self.bn = BatchNorm(num_units, True, is_mask)
        if self.with_pre_norm:
            self.layerNorm = nn.LayerNorm(num_units)
        self.dropout = nn.Dropout(p=1 - self.keep_prob)

    def forward(self, seq, mask):
        if self.with_pre_norm:
            seq = self.layerNorm(seq.transpose(1, 2)).transpose(1, 2)
        in_c = seq.size()[1]
        seq = torch.transpose(seq, 1, 2)  # (N, L, C)
        queries = seq
        keys = seq
        num_heads = self.num_heads

        # T_q = T_k = L
        Q = self.linear_q(seq)  # (N, T_q, C)
        K = self.linear_k(seq)  # (N, T_k, C)
        V = self.linear_v(seq)  # (N, T_k, C)

        # Split and concat
        Q_ = torch.cat(torch.split(Q, in_c // num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        # Key Masking
        key_masks = mask.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1)  # (h*N, 1, T_k)
        key_masks = key_masks.repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-INF)  # extremely small value
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)

        query_masks = mask.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = query_masks.repeat(1, 1, keys.size()[1]).float()  # (h*N, T_q, T_k)

        att_scores = F.softmax(outputs, dim=-1) * query_masks  # (h*N, T_q, T_k)
        att_scores = self.dropout(att_scores)

        # Weighted sum
        x_outputs = torch.matmul(att_scores, V_)  # (h*N, T_q, C/h)
        # Restore shape
        x_outputs = torch.cat(
            torch.split(x_outputs, x_outputs.size()[0] // num_heads, dim=0),
            dim=2)  # (N, T_q, C)

        # transform for the output
        x_outputs = self.o_net(x_outputs)

        x = torch.transpose(x_outputs, 1, 2)  # (N, C, L)
        if self.with_bn:
            x = self.bn(x, mask)

        return x

class Attention_old(nn.Module):
    def __init__(self, num_units, num_heads, keep_prob, is_mask):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.keep_prob = keep_prob

        self.linear_q = nn.Linear(num_units, num_units)
        self.linear_k = nn.Linear(num_units, num_units)
        self.linear_v = nn.Linear(num_units, num_units)

        self.bn = BatchNorm(num_units, True, is_mask)
        self.dropout = nn.Dropout(p=1 - self.keep_prob)

    def forward(self, seq, mask):
        in_c = seq.size()[1]
        seq = torch.transpose(seq, 1, 2)  # (N, L, C)
        queries = seq
        keys = seq
        num_heads = self.num_heads

        # T_q = T_k = L
        Q = F.relu(self.linear_q(seq))  # (N, T_q, C)
        K = F.relu(self.linear_k(seq))  # (N, T_k, C)
        V = F.relu(self.linear_v(seq))  # (N, T_k, C)

        # Split and concat
        Q_ = torch.cat(torch.split(Q, in_c // num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        # Key Masking
        key_masks = mask.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1)  # (h*N, 1, T_k)
        key_masks = key_masks.repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-INF)  # extremely small value
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)

        query_masks = mask.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = query_masks.repeat(1, 1, keys.size()[1]).float()  # (h*N, T_q, T_k)

        att_scores = F.softmax(outputs, dim=-1) * query_masks  # (h*N, T_q, T_k)
        att_scores = self.dropout(att_scores)

        # Weighted sum
        x_outputs = torch.matmul(att_scores, V_)  # (h*N, T_q, C/h)
        # Restore shape
        x_outputs = torch.cat(
            torch.split(x_outputs, x_outputs.size()[0] // num_heads, dim=0),
            dim=2)  # (N, T_q, C)

        x = torch.transpose(x_outputs, 1, 2)  # (N, C, L)
        x = self.bn(x, mask)

        return x


class RNN(nn.Module):
    def __init__(self, hidden_size, output_keep_prob, with_pre_norm):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.bid_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_keep_prob = output_keep_prob

        self.out_dropout = nn.Dropout(p=(1 - self.output_keep_prob))

        self.with_pre_norm = with_pre_norm
        if self.with_pre_norm:
            self.layerNorm = nn.LayerNorm(hidden_size)

    def forward(self, seq, mask):
        # seq: (N, C, L)
        # mask: (N, L)
        if self.with_pre_norm:
            seq = self.layerNorm(seq.transpose(1, 2)).transpose(1, 2)
        max_len = seq.size()[2]
        length = get_length(mask).cpu()
        seq = torch.transpose(seq, 1, 2)  # to (N, L, C)
        packed_seq = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True,
                                                       enforce_sorted=False)
        outputs, _ = self.bid_rnn(packed_seq)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                   total_length=max_len)[0]
        outputs = outputs.view(-1, max_len, 2, self.hidden_size).sum(2)  # (N, L, C)
        outputs = self.out_dropout(outputs)  # output dropout
        return torch.transpose(outputs, 1, 2)  # back to: (N, C, L)


class LinearCombine(nn.Module):
    def __init__(self, layers_num, trainable=True, input_aware=False, word_level=False):
        super(LinearCombine, self).__init__()
        self.input_aware = input_aware
        self.word_level = word_level

        if input_aware:
            raise NotImplementedError("Input aware is not supported.")
        self.w = nn.Parameter(torch.full((layers_num, 1, 1, 1), 1.0 / layers_num),
                              requires_grad=trainable)

    def forward(self, seq):
        nw = F.softmax(self.w, dim=0)
        seq = torch.mul(seq, nw)
        seq = torch.sum(seq, dim=0)
        return seq

def get_param_map(dim = 256, att_head = 8, with_pre_norm = True, with_bn = False, print_func = lambda x: None):

    drop_prob = 0.1
    def conv_shortcut(kernel_size):
        return ConvBN(kernel_size, dim, dim, 1 - drop_prob, False, True, with_pre_norm=with_pre_norm, with_bn=with_bn)
    
    def get_edge_module(index):
        if 0 == index:
            return conv_shortcut(1)
        if 1 == index:
            return Attention(dim, att_head, 1 - drop_prob, True, with_bn=with_bn, with_pre_norm=with_pre_norm)
        if 4 == index:
            return conv_shortcut(3)
        if 5 == index:
            return conv_shortcut(5)
        if 6 == index:
            return conv_shortcut(7)
        if 7 == index:
            return AvgPool(3, False, True, with_pre_norm=with_pre_norm, dim=dim)
        if 8 == index:
            return MaxPool(3, False, True, with_pre_norm=with_pre_norm, dim=dim)
        if 9 == index:
            return RNN(dim, 1 - drop_prob, with_pre_norm=with_pre_norm)
    
    param = {}

    a = get_edge_module(0)
    print_func("conv - 1: param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[0] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(4)
    print_func("conv - 3: param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[4] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(5)
    print_func("conv - 5: param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[5] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(6)
    print_func("conv - 7: param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[6] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(1)
    print_func("attn    : param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[1] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(9)
    print_func("rnn     : param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[9] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(7)
    print_func("avg pool: param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[7] = sum([x.nelement() for x in a.parameters()])
    a = get_edge_module(8)
    print_func("max pool: param: %d" % (sum([x.nelement() for x in a.parameters()])))
    param[8] = sum([x.nelement() for x in a.parameters()])
    
    return param

class Zero(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x, *args, **kwargs):
        return self.dummy * x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def forward(self, x, *args, **kwargs):
        return self.dummy * x

OPS = {
    'ZERO': lambda dim, dropout, act=None, norm=None, pre=True: Zero(),
    'IDEN': lambda dim, dropout, act=None, norm=None, pre=True: Identity(),
    'CONV1': lambda dim, dropout, act=None, norm=None, pre=True: ConvBN(1, dim, dim, 1 - dropout, False, True, False, act==nn.ReLU, norm=='ln'),
    'CONV3': lambda dim, dropout, act=None, norm=None, pre=True: ConvBN(3, dim, dim, 1 - dropout, False, True, False, act==nn.ReLU, norm=='ln'),
    'MAX': lambda dim, dropout, act=None, norm=None, pre=True: MaxPool(3, False, True, norm == 'ln', dim),
    'GRU': lambda dim, dropout, act=None, norm=None, pre=True: RNN(dim, 1 - dropout, norm=='ln')
}

PRIMITIVES = list(OPS.keys())
