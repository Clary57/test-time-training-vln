import os
import argparse
import json
import math

def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
)
args = parser.parse_args()

jsons = os.listdir(os.path.join(args.path, 'log'))
acc = 0
total_count = 0
for j in jsons:
    with open(os.path.join(args.path, 'log', j)) as f:
        try:
            data = json.load(f)
            acc += check_inf_nan(int(data['acc']))
            total_count += check_inf_nan(data['total_count'])
        except:
            print(j)
print(f'acc: {acc:.4f}/{total_count} ({acc/total_count:.4f})')
