import re
import numpy as np

number_pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')


def is_number(s):
    result = number_pattern.match(s)
    if result:
        return True
    else:
        return False


def read_dataset(path):
    dataset = []
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            data = []
            for s in line:
                if is_number(s):
                    data.append(float(s))
                else:
                    if s not in mapping:
                        mapping[s] = float(len(mapping))
                    data.append(mapping[s])
            dataset.append(data)
    result = np.zeros((len(dataset), len(dataset[0])), dtype=np.float)
    i = 0
    for data in dataset:
        j = 0
        for d in data:
            result[i][j] = d
            j += 1
        i += 1
    return result
