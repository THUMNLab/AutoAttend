from copy import deepcopy
import random

def remove(source, element):
    while element in source:
        source.remove(element)
    return source

def check_valid(arch, primitives):

    zero = primitives.index('ZERO')

    for i, a in enumerate(arch):
        o1, prev, o2, o3, n = a
        if prev == -1:
            return False
        if n == 1 and (o2 == zero or o3 == zero):
            return False
        if o1 == zero:
            return False
        if (o2 == -1 or o3 == -1) and n == 1:
            return False

    return True

def sample_valid_arch(node_num, edgeop, nodeop, primitives):
    arch = []
    zero = primitives.index('ZERO')
    for i in range(node_num):
        edges = deepcopy(edgeop)
        idx_pool = list(range(i + 1))
        n = random.choice(nodeop)
        prev = random.choice(idx_pool)
        edge_no_zero = deepcopy(edgeop)
        remove(edge_no_zero, zero)
        
        o1 = random.choice(edge_no_zero)
        if n == 1:
            o2 = random.choice(edge_no_zero)
            o3 = random.choice(edge_no_zero)
        else:
            o2 = random.choice(edgeop)
            o3 = -1
        arch.append([o1, prev, o2, o3, n])
    if check_valid(arch, primitives):
        return arch
    return sample_valid_arch(node_num, edgeop, nodeop, primitives)

def sample_valid_archs(node_num, edgeop, nodeop, number, primitives):
    total_arch = []
    while len(total_arch) < number:
        arch = sample_valid_arch(node_num, edgeop, nodeop, primitives)
        if arch not in total_arch:
            total_arch.append(arch)
    return total_arch

def get_edge_node_op(primitives, space=0):
    # no att
    if space == 0:
        return list(range(len(primitives))), [0]
    # att
    elif space == 1:
        return list(range(len(primitives))), [0, 1]

def reduce(arch):
    new_arch = deepcopy(arch)
    for i in range(len(arch)):
        if new_arch[i][-1] == 0:
            new_arch[i][-2] = -1
    return new_arch

def mutate_arch(arch, edgeops, nodeops, primitives ,ratio=0.05):
    new_arch = deepcopy(arch)
    zero = primitives.index('ZERO')
    iden = primitives.index('IDEN')
    edge_no_zero = deepcopy(edgeops)
    remove(edge_no_zero, zero)

    for i in range(len(arch)):
        o1, prev, o2, o3, n = new_arch[i]
        mutate_all = False

        # mutate node
        if random.random() < ratio and len(nodeops) > 1:
            n = 1 - n
            mutate_all = True

        # mutate prev
        if mutate_all or random.random() < ratio:
            prev = random.choice(list(range(i + 1)))

        # mutate o1
        if mutate_all or random.random() < ratio:
            o1 = random.choice([x for x in edgeops if x != zero])
        
        # mutate o2
        if mutate_all or random.random() < ratio:
            o2 = random.choice(edgeops if n == 0 else edge_no_zero)
        
        # mutate o3
        if mutate_all or random.random() < ratio:
            o3 = -1 if n == 0 else random.choice(edge_no_zero)
        
        new_arch[i] = [o1, prev, o2, o3, n]

    if new_arch == reduce(arch) or not check_valid(new_arch, primitives):
        return mutate_arch(arch, edgeops, nodeops, primitives, ratio)
    return new_arch

if __name__ == '__main__':
    import os
    import importlib
    import argparse
    from tqdm import tqdm

    import torch

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--space', type=int, default=1, help='search space')
    parser.add_argument('--number', type=int, help='number of archs to spawn')
    parser.add_argument('--unique', action='store_true', help='whether spawned arch can overlap')
    parser.add_argument('--repeat', type=int, default=1, help='repeat how many time of sampled arch')
    parser.add_argument('--path', type=str, help='path to save spawned archs')
    parser.add_argument('--node', type=int, default=2, help='number of layers')
    parser.add_argument('--primitive', type=str, default='model.sent.arch', help='primitive archs pool')

    args = parser.parse_args()
    primitives = getattr(importlib.import_module(args.primitive), 'PRIMITIVES')

    edgeop, nodeop = get_edge_node_op(primitives, args.space)
    if args.unique:
        arch = sample_valid_archs(args.node, edgeop, nodeop, args.number, primitives)
    else:
        arch = [sample_valid_arch(args.node, edgeop, nodeop, primitives) for _ in tqdm(range(args.number))]
    
    arch *= args.repeat

    # make dirs
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    torch.save(arch, args.path)
