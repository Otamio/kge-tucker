import glob
from collections import namedtuple
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='')
parser.add_argument("--model", type=str, default='')
parser.add_argument("--mode", type=str, default='')
parser.add_argument("--path", type=str, default='out')
Result = namedtuple("Result", "mr mrr hits_1 hits_3 hits_10")


class EpocResult(namedtuple("EpocResult", "epoc validation test")):
    def __gt__(self, other):
        return self.validation.mrr > other.validation.mrr

    def __lt__(self, other):
        return self.validation.mrr < other.validation.mrr

    def __eq__(self, other):
        return self.validation.mrr == other.validation.mrr


def convert_result(dic):
    return Result(dic['mr'],
                  dic['mrr'],
                  dic['hits@1'],
                  dic['hits@3'],
                  dic['hits@10'])


def main():

    args = parser.parse_args()
    results = {}
    for fname in glob.iglob(f"{args.path}/*{args.dataset}*{args.mode}*{args.model}*.log"):
        try:
            with open(fname) as fd:
                epoc = 0
                result = []
                valid, test = {}, {}
                current = valid
                for line in fd:
                    line = line.split('INFO')[1].strip()
                    if line.startswith("Validation at"):
                        if len(test) > 0:
                            result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
                        valid, test = {}, {}
                        current = valid
                        epoc = int(line.split()[-1])
                    if line.startswith("Test at"):
                        current = test
                    elif line.startswith("Hits @10"):
                        current["hits@10"] = float(line.split(':')[1])
                    elif line.startswith("Hits @3"):
                        current["hits@3"] = float(line.split(':')[1])
                    elif line.startswith("Hits @1"):
                        current["hits@1"] = float(line.split(':')[1])
                    elif line.startswith("Mean rank"):
                        current["mr"] = float(line.split(':')[1])
                    elif line.startswith("Mean reciprocal rank"):
                        current["mrr"] = float(line.split(':')[1])
                result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
            # if epoc % 100 == 0:
            results[fname.split('/')[1].split('.')[0].strip()] = result
            # else:
                #print('Running:', fname, epoc)
        except KeyError as e:
            print('Running:', fname, e)
            pass

    frame = []
    for exp, res in sorted(results.items()):
        best = max(res)
        frame.append({
            "experiment": exp,
            "best_epoc": best.epoc,
            "mrr": round(best.test.mrr, 3),
            "hits@1": round(best.test.hits_1, 3),
            "hits@10": round(best.test.hits_10, 3)
        })
    print(pd.DataFrame(frame).sort_values(by=["mrr", "hits@1", "hits@10"], ascending=False))


if __name__ == "__main__":
    main()
