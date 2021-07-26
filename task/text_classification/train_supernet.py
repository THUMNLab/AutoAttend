'''
train the supernet
'''

import numpy as np
from model.ops import PRIMITIVES
import os
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.utils import get_edge_node_op, sample_valid_archs
from model.supernet import TextClassifier
from task.utils import logger, set_seed

LOGGER = logger.get_logger('sst5-search')

if __name__ == '__main__':

    logger.LEVEL = logger.INFO
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dim', type=int, help='dimension', default=64)
    parser.add_argument('--head', type=int, help='default attn head number', default=8)
    parser.add_argument('--layer', type=int, help='total layer number', default=24)
    parser.add_argument('--space', type=int, help='search space type', default=1)
    parser.add_argument('--context', choices=['sc', 'fc', 'tc', 'nc'], default='fc')
    parser.add_argument('--dataset', type=str, help='path to dataset', default='./data')

    parser.add_argument('--arch_batch_size', type=int, help='batch size to train the base', default=16)
    parser.add_argument('--epoch', type=int, help='total num of epoch to train the supernet', default=10)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.1)
    parser.add_argument('--gradient_clip', type=float, help='gradient clip', default=5.0)
    
    parser.add_argument('--path', type=str, help='path to save logs & models', default='./searched')

    parser.add_argument('--seed', type=int, help='random seed', default=2021)
    parser.add_argument('--device', type=int, default=0, help='the main progress')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.path is not None:
        os.makedirs(args.path, exist_ok=True)
        LOGGER.set_path(os.path.join(args.path, 'log.txt'))

    LOGGER.info('hyper parameters')
    for k,v in args.__dict__.items():
        LOGGER.info('{} - {}'.format(k, v))
    LOGGER.info('end of hyper parameters')

    input('hyperparameter confirmed')

    LOGGER.info('load dataset...')
    edgeop, nodeop = get_edge_node_op(PRIMITIVES, args.space)
    train_dataset = torch.load(os.path.join(args.dataset, 'sst/train.pt'))
    valid_dataset = torch.load(os.path.join(args.dataset, 'sst/valid.pt'))
    test_dataset  = torch.load(os.path.join(args.dataset, 'sst/test.pt'))
    embedding = torch.load(os.path.join(args.dataset, 'sst/embedding.pt'))
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    device = torch.device('cuda:{}'.format(args.device))

    LOGGER.info('build model...')
    model = TextClassifier(
        embeddings=embedding,
        dim=args.dim,
        head=args.head,
        nclass=5,
        layer=args.layer,
        edgeops=edgeop,
        nodeops=nodeop,
        dropout=args.dropout,
        context=args.context, aug_dropouts=[0.1, 0.1]).to(device)

    LOGGER.info('train supernet...')
    model.train()
    eval_id = 0
    time_list = []
    loss_item = []

    opt = torch.optim.Adam(model.parameters(), lr=args.lr / args.arch_batch_size, weight_decay=args.weight_decay)

    for idx in range(args.epoch):
        t = tqdm(train_dataloader)
        for batch in t:
            opt.zero_grad()
            
            for arch in sample_valid_archs(args.layer, edgeop, nodeop, args.arch_batch_size, PRIMITIVES):                
                logit = model(batch[0].to(device), batch[2].to(device), arch)
                loss = F.cross_entropy(logit, batch[1].long().to(device))
                loss.backward()
                loss_item.append(loss.item())
                loss_item = loss_item[-100:]

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
            
            opt.step()
            t.set_postfix(loss=round(np.mean(loss_item), 4))
        if args.path:
            torch.save(model, os.path.join(args.path, 'model_epoch_{}.full'.format(idx)))
