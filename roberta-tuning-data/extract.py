import os
import json
from collections import defaultdict
import pandas as pd

results = defaultdict(list)


# directory = '../train_model_2024-04-09_16-33-58'
directory = 'INSERT DIRECTORY PATH HERE'

for filename in os.listdir(directory):
    inner_directory = os.path.join(directory, filename)
    vals = {}
    loss = []
    if not os.path.isdir(inner_directory):
        continue
    for inner_filename in os.listdir(inner_directory):
        if inner_filename == 'params.json':
            with open(os.path.join(inner_directory, inner_filename)) as f:
                try:
                    data = json.load(f)
                except:
                    continue
                vals['lr'] = data['lr']
                vals['weight_decay'] = data['weight_decay']
                vals['train_batch_size'] = data['train_batch_size']
                vals['optimizer'] = data['optimizer']
        if inner_filename == 'progress.csv':
            with open(os.path.join(inner_directory, inner_filename)) as f:
                try:
                    data = pd.read_csv(f)
                    # take the last row
                    data = data.iloc[-1].to_dict()
                except:
                    continue
                vals['val_loss'] = data['val_loss']
                vals['loss'] = data['loss']
                vals['accuracy'] = data['accuracy']
    if 'loss' not in vals or 'lr' not in vals:
        continue
    param_tup = (vals['lr'], vals['weight_decay'], vals['train_batch_size'], vals['optimizer'])
    param_str = f"lr={vals['lr']}, weight_decay={vals['weight_decay']}, train_batch_size={vals['train_batch_size']}, optimizer={vals['optimizer']}"
    results[param_str].append(f"loss={vals['loss']}, val_loss={vals['val_loss']}, accuracy={vals['accuracy']}")
    # results.append(vals)

# write to json file
print(results)
with open('tuning-data-filtered.json', 'w') as f:
    json.dump(results, f)