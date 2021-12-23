import glob
from collections import namedtuple

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

    results = {}

    for fname in glob.iglob("out/*.log"):
        with open(fname) as fd:
            epoc = 0
            result = []
            cache = {}
            for line in fd:
                line = line.strip()
                if line.startswith("Iteration:"):
                    epoc = int(line.split(':')[1])
                elif len(cache) > 0 and \
                        (line.startswith("Validation:") or line.startswith("Test:")):
                    if line.startswith("Validation:"):
                        result[-1] = EpocResult(epoc, result[-1], convert_result(cache))
                    elif line.startswith("Test:"):
                        result.append(convert_result(cache))
                    cache = {}
                elif line.startswith("Hits @10"):
                    cache["hits@10"] = float(line.split(':')[1])
                elif line.startswith("Hits @3"):
                    cache["hits@3"] = float(line.split(':')[1])
                elif line.startswith("Hits @1"):
                    cache["hits@1"] = float(line.split(':')[1])
                elif line.startswith("Mean rank"):
                    cache["mr"] = float(line.split(':')[1])
                elif line.startswith("Mean reciprocal rank"):
                    cache["mrr"] = float(line.split(':')[1])
            result[-1] = EpocResult(epoc, result[-1], convert_result(cache))
        results[fname.split('/')[1].split('.')[0].strip()] = result

    for exp, res in results.items():
        best = max(res)
        print(exp, best.epoc, '\t', best.test.mrr, '\t', best.test.hits_1, '\t', best.test.hits_10)


if __name__ == "__main__":
    main()
