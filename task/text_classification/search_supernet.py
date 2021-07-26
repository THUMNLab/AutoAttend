'''
search for best models given supernet using evolution algorithms
'''

import os

import numpy as np
from tqdm import tqdm
from queue import Queue
import threading

import torch
from torch.utils.data import DataLoader

from model.utils import get_edge_node_op, sample_valid_archs, mutate_arch
from model.supernet import PRIMITIVES
from ..utils import run_exps

def wrap_queue(lists):
    queue = Queue()
    for ele in lists: queue.put(ele)
    queue.put(False)
    return queue

def eval_model(model, arch, loaders, device):
    model.eval()
    gt = []
    pred = []
    with torch.no_grad():
        for batch in loaders:
            logit = model(batch[0].to(device), batch[2].to(device), arch)
            pred.extend(logit.argmax(1).detach().cpu().tolist())
            gt.extend(batch[1].cpu().tolist())
    return (np.array(gt) == np.array(pred)).mean()


def eval_param_archs(model_path, archs, device, mask='val', output=None):
    if mask == 'val':
        loader = torch.load('data/sst/valid.pt')
    else:
        loader = torch.load('data/sst/test.pt')
    loader = DataLoader(loader, batch_size=512, shuffle=False)
    
    perf = []
    # print('load model')
    model = torch.load(model_path, map_location=device)

    # print('testing valid scores')
    for a in archs:
        score = eval_model(model, a, loader, device)
        perf.append([a, score])

    return {
        'perf': perf,
        'pid': os.getpid(),
    }

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--init_pop', type=int, default=500)
    parser.add_argument('--mutate_number', type=int, default=100)
    parser.add_argument('--mutate_epoch', type=int, default=5)
    parser.add_argument('--devices', type=int, nargs='+')
    parser.add_argument('--output', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--layer', type=int, default=24)
    parser.add_argument('--space', type=int, default=6)
    parser.add_argument('--chunk', type=int, default=25)

    args = parser.parse_args()

    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.model_path):
        name = os.listdir(args.model_path)
        name = [os.path.join(args.model_path, n) for n in name if n.startswith('model_') and n.endswith('.full')]
    else:
        name = [args.model_path]

    arch2performance = {}
    devices = Queue()
    for device in args.devices: devices.put(f'cuda:{device}')
    rlock = threading.Lock()

    progress = tqdm(total=args.init_pop + args.mutate_epoch * args.mutate_number)

    progress.set_description('initial')

    # init population
    def process_result(res):
        if res is None: return
        os.system(f'kill -9 {res["pid"]}')
        rlock.acquire()
        for line in res['perf']:
            arch2performance[str(line[0])] = line[1]
            progress.update(1)
        rlock.release()

    edgeop, nodeop = get_edge_node_op(PRIMITIVES, args.space)
    archs = sample_valid_archs(args.layer, edgeop, nodeop, args.init_pop, PRIMITIVES)

    archs_passed = [[]]
    for a in archs:
        if len(archs_passed[-1]) == args.chunk:
            archs_passed.append([])
        archs_passed[-1].append(a)

    run_exps(devices, wrap_queue([{
        'func': eval_param_archs,
        'kwargs': dict(model_path=args.model_path, archs=a),
        'callback': process_result
    } for a in archs_passed]))

    for i in range(args.mutate_epoch):
        progress.set_description(f'epoch: {i}')

        # mutate architectures
        current_archs = list(arch2performance.items())
        current_archs = sorted(current_archs, key=lambda x:-x[1])
        mutated = current_archs[:args.mutate_number]
        arch_new = [[]]
        for arch in mutated:
            arch = eval(arch[0])
            a = mutate_arch(arch, edgeop, nodeop, PRIMITIVES)
            while str(a) in arch2performance: a = mutate_arch(arch, edgeop, nodeop, PRIMITIVES)
            if len(arch_new[-1]) == args.chunk: arch_new.append([])
            arch_new[-1].append(a)
        
        # run jobs
        run_exps(devices, wrap_queue([{
            'func': eval_param_archs,
            'kwargs': dict(model_path=args.model_path, archs=a),
            'callback': process_result
        } for a in arch_new]))
    
    # derive final lists
    archs = sorted(list(arch2performance.items()), key=lambda x:-x[1])
    torch.save([[eval(x[0]), x[1]] for x in archs], os.path.join(args.output, 'performance.dict'))
