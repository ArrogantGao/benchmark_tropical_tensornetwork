import json, math
import torch
import time
from copy import deepcopy

def myeinsum(ixs, iy, tensors):
    uniquelabels = list(set(sum(ixs, start=[]) + iy))
    allow_ascii = list(range(65, 90)) + list(range(97, 122))
    labelmap = {l:chr(allow_ascii[i]) for i, l in enumerate(uniquelabels)}
    eins = ",".join(["".join([labelmap[l] for l in ix]) for ix in ixs]) + "->" + "".join([labelmap[l] for l in iy])
    return torch.einsum(eins, *tensors)

def contract_recur(tree:dict, inputs):
    if "args" in tree:
        tensors = [contract_recur(arg, inputs) for arg in tree["args"]]
        return myeinsum(tree["eins"]["ixs"], tree["eins"]["iy"], tensors)
    else:
        return inputs[tree["tensorindex"]-1]

def contract(tree:dict, inputs):
    labels = torch.unique(torch.tensor(sum(tree["inputs"], start=[]) + tree["output"]))
    return contract_recur(tree['tree'], inputs)

def gpu(repeat_times = 10, tensornetwork="../networks/sc31/eincode_1.json", deviceid:int=0):
    device = 'cuda:%d'%deviceid
    with open(tensornetwork, 'r') as f:
        optcode = json.load(f)

    
    tensors = []
    for ix in optcode["inputs"]:
        if len(ix) == 2:
            t = torch.zeros((2, 2), dtype=torch.float32, device=device)
            t[1, 1] = float('-inf')
        else:
            t = torch.zeros((2), dtype=torch.float32, device=device)
            t[1] = float(1.0)
        tensors.append(t)
    
    # print(tensors)

    torch.cuda.synchronize(device)
    ta = time.time()
    mintime = math.inf
    for _ in range(repeat_times):
        t0 = time.time()
        res = contract(optcode, tensors)
        torch.cuda.synchronize(device)
        t1 = time.time()
        mintime = min(mintime, t1-t0)
        print(res)
    total = time.time() - ta
    print("minimum time = \n", mintime)

gpu()
