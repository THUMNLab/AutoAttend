'''
retrain models on sst5
'''

import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.supernet import TextClassifier
from task.utils import logger, set_seed
from .search_supernet import eval_model

LOGGER = logger.get_logger('sst5-retrain')

def retrain_arch_mini_batch(dim, head, arch, context, lr, wd, dr, repeat, device, path, epoch, eval_iter=-1, patience=-1, gradient_clip=1.0, dropout_attn=0.0, dropout_aggr=0.1):
    log = logger.get_logger('sst5-retrain')
    if path:
        log.set_path(path.replace('.pt', '.log'))
    train_dataset = torch.load('data/sst/train.pt')
    valid_dataset = torch.load('data/sst/valid.pt')
    test_dataset  = torch.load('data/sst/test.pt')
    embedding = torch.load('data/sst/embedding.pt')
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    vals, tests = [], []
    for r in range(repeat):
        model = TextClassifier(embedding, dim, head, 5, arch=arch, dropout=dr, context=context, aug_dropouts=[dropout_attn, dropout_aggr]).to(device)
        val_acc = []
        test_acc = []
        losses = []
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        iter = 0
        for e in range(epoch):
            with tqdm(train_dataloader) as t:
                for batch in t:
                    iter += 1
                    model.train()
                    opt.zero_grad()
                    logit = model(batch[0].to(device), batch[2].to(device))
                    loss = F.cross_entropy(logit, batch[1].to(device).long())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    opt.step()
                    losses = (losses + [loss.item()])[-100:]
                    t.set_postfix(loss='%.4f' % (np.mean(losses)), val=0.0 if val_acc == [] else '%.4f' % max(val_acc), test=0.0 if test_acc == [] else '%.4f' % max(test_acc))

                    if eval_iter >= 2 and iter % eval_iter == 0:
                        model.eval()
                        val = eval_model(model, None, valid_dataloader, device)
                        test = eval_model(model, None, test_dataloader, device)
                        val_acc.append(val)
                        test_acc.append(test)
                        # judge whether patience run out
                        if patience > 0:
                            idx_max = np.argmax(val_acc)
                            log.info('epoch', e, 'iter', iter, 'val / test', val, test, 'patience', len(val_acc) - idx_max)
                            if len(val_acc) - idx_max > patience:
                                break
                        else:
                            log.info('epoch', e, 'iter', iter, 'val / test', val, test)

            
            if eval_iter <= 0:
                model.eval()
                val = eval_model(model, None, valid_dataloader, device)
                test = eval_model(model, None, test_dataloader, device)
                val_acc.append(val)
                test_acc.append(test)
                # judge whether patience run out
                if patience > 0:
                    idx_max = np.argmax(val_acc)
                    log.info('epoch', e, 'val / test', val, test, 'patience', len(val_acc) - idx_max)
                    if len(val_acc) - idx_max > patience:
                        break
                else:
                    log.info('epoch', e, 'val / test', val, test)
            
        # record this repeat
        vals.append(val_acc)
        tests.append(test_acc)

    info = {
        'mode': 'mini',
        'arch': arch,
        'val': vals,
        'test': tests,
        'config': {
            'dim': dim,
            'head': head,
            'context': context, 
            'lr': lr, 
            'wd': wd, 
            'dr': dr, 
            'eval_iter': eval_iter,
            'patience': patience,
            'repeat': repeat, 
            'device': device
        }
    }
    
    if path:
        torch.save(info, path)
    
    return info

def retrain_arch(dim, head, arch, context, lr, wd, dr, repeat, device, path, epoch, eval_iter=-1, patience=-1, gradient_clip=1.0, dropout_attn=0.0, dropout_aggr=0.1):
    train_dataset = torch.load('data/sst/train.pt')
    valid_dataset = torch.load('data/sst/valid.pt')
    test_dataset  = torch.load('data/sst/test.pt')
    embedding = torch.load('data/sst/embedding.pt')
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    vals, tests = [], []
    for r in range(repeat):
        model = TextClassifier(embedding, dim, head, 5, arch=arch, dropout=dr, context=context, aug_dropouts=[dropout_attn, dropout_aggr]).to(device)
        val_acc = []
        test_acc = []
        losses = []
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        for e in range(epoch):
            model.train()
            with tqdm(train_dataloader) as t:
                for batch in t:
                    opt.zero_grad()
                    logit = model(batch[0].to(device), batch[2].to(device))
                    loss = F.cross_entropy(logit, batch[1].to(device).long())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    opt.step()
                    losses = (losses + [loss.item()])[-100:]
                    t.set_postfix(loss='%.4f' % (np.mean(losses)))
            
            model.eval()
            val = eval_model(model, None, valid_dataloader, device)
            test = eval_model(model, None, test_dataloader, device)
            val_acc.append(val)
            test_acc.append(test)
            print('epoch', e, 'val:', val, 'test:', test)

        # record this repeat
        vals.append(val_acc)
        tests.append(test_acc)

    info = {
        'arch': arch,
        'val': vals,
        'test': tests,
        'config': {
            'dim': dim,
            'head': head,
            'context': context, 
            'lr': lr, 
            'wd': wd, 
            'dr': dr, 
            'repeat': repeat, 
            'device': device
        }
    }
    
    if path:
        torch.save(info, path)
    
    return info

if __name__ == '__main__':

    logger.LEVEL = logger.DEBUG
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dim', type=int, default=256, help='dimension')
    parser.add_argument('--head', type=int, default=8, help='default attn head number')
    parser.add_argument('--arch', type=str, help='architectures (str)')
    parser.add_argument('--context', choices=['sc', 'fc', 'tc', 'nc'], default='fc')

    parser.add_argument('--epoch', type=int, help='total num of epoch to train the supernet', default=10)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.0)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.1)
    parser.add_argument('--repeat', type=int, default=5)
    
    parser.add_argument('--path', type=str, help='path to save logs & models', default='logs')

    parser.add_argument('--seed', type=int, help='random seed', default=2021)
    parser.add_argument('--no_mini', action='store_true', help='wether to disable mini-batch mode')
    parser.add_argument('--patience', type=int, help='patience', default=-1)
    parser.add_argument('--eval_iter', type=int, help='eval iteration', default=100)
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='the main progress')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='gradient clip')
    parser.add_argument('--dropout_attn', type=float, default=0.1, help='dropout applied to attention mask')
    parser.add_argument('--dropout_aggr', type=float, default=0.1, help='dropout applied after attention aggregation')

    args = parser.parse_args()
    args.mini = not args.no_mini

    set_seed(args.seed)

    if args.path is not None:
        os.makedirs(args.path, exist_ok=True)
        LOGGER.set_path(os.path.join(args.path, 'log.txt'))

    LOGGER.info('hyper parameters')
    for k,v in args.__dict__.items():
        LOGGER.info('{} - {}'.format(k, v))
    LOGGER.info('end of hyper parameters')

    input('hyperparameter confirmed')
    func = retrain_arch_mini_batch if args.mini else retrain_arch

    device = 'cuda:{}'.format(args.device[0])
    if args.path is None: path_model = None
    else:
        path_model = os.path.join(
            args.path,
            f'{args.dim}_{args.head}_{args.epoch}_{args.lr}_{args.weight_decay}_{args.dropout}_{args.gradient_clip}_{args.dropout_attn}_{args.dropout_aggr}.pt'
        )

    result = func(
        args.dim, args.head, eval(args.arch), args.context, args.lr, args.weight_decay, args.dropout, args.repeat, device,
        path_model, args.epoch, args.eval_iter, args.patience, args.gradient_clip, args.dropout_attn, args.dropout_aggr,
    )

    print(result)
    print('final test', sum(result['test']) / args.repeat)
