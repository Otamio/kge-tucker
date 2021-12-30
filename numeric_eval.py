import argparse
import os
import json
import torch
from load_data import Data
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import DistMult, ComplEx, ConvE, TuckER


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--model', default='rotate', help='Please provide a model to run')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')


def populate_estimate(model, test, suffix):
    ent, rel = [], []
    for i, row in test.iterrows():
        ent.append(ent2idx[row[0]])
        rel.append(rel2idx["Interval-" + row[1] + suffix])
    res = model(torch.LongTensor(ent), torch.LongTensor(rel))

    for i, row in tqdm(test.iterrows()):
        try:
            candidates = list(filter(
                lambda x: row[1] in x,
                list(map(lambda x: idx2ent[x], [x.item() for x in torch.argsort(res[i], descending=True)[:50]]))
            ))
            test[suffix][i] = medians[candidates[0]]
        except KeyError:
            print(i, row[0], row[1])
            test[suffix][i] = medians[row[1]]


def compute_result(test):
    res = []
    for p in test[1].unique():
        sli = test[test[1] == p]
        res.append({
            "Property": p,
            "MAE": sli.iloc[:, -1].mean()
        })
    print(pd.DataFrame(res))


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    target = f'numeric/{args.dataset}'

    with open(f'{target}/{args.model}_entities.dict') as fd:
        ent2idx = json.load(fd)
    with open(f'{target}/{args.model}_relations.dict') as fd:
        rel2idx = json.load(fd)
    idx2ent = {v: k for k, v in ent2idx.items()}
    with open(f'{target}/medians.dict') as fd:
        medians = json.load(fd)

    d = Data(data_dir=f"{target}/", reverse=True)
    if args.model == "distmult":
        model = DistMult(d, 200, 200, **{"input_dropout": 0.3})
    else:
        print("Unsupported Model", args.model)
        exit()

    model.load_state_dict(torch.load(f"{target}/{args.model}.model"))

    if 'QOC' in args.dataset or 'FOC' in args.dataset:
        runs = ["_left", "_right"]
    else:
        exit()

    test = pd.read_csv(f'{target}/test_raw.txt', sep='\t', header=None)
    test.columns = ["node", "label", "value"]
    for suffix in runs:
        test[suffix] = np.nan
        populate_estimate(model, test, suffix)

    test["estimate"] = 0
    for suffix in runs:
        test["estimate"] += test[suffix]
    test["estimate"] /= len(runs)

    compute_result(test)
