import os
import pandas as pd
from pathlib import Path
import json
from collections import Counter

df = pd.read_csv('classifier_pred_on_selection.csv')
df.head()
print(df.columns.values)

accents = list(df.columns.values)[:-4]
budget=200
budget_dur=int(540/100*budget)
stats_dict = {}
for accent in accents:
    df.sort_values(accent, ascending=False, inplace=True)
    for i in range(1, 4):
        path = f'/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/{accent}/manifests/TSS_output/all/budget_{budget}/classifier/run_{i}/train/'
        Path(path).mkdir(parents=True, exist_ok=True)
        stats = []
        with open(path+'train.json', 'w') as fp:
            total_dur, idx = 0, 0
            while total_dur < budget_dur:
                row = df.iloc[idx]
                row_dict = {
                        'text': row['text'],
                        'audio_filepath': row['audio_filepath'],
                        'duration': row['duration'],
                        'dir': row['dir']
                        }
                stats.append(row['dir'])
                json.dump(row_dict, fp)
                fp.write('\n')
                idx += 1
                total_dur += row['duration']
            stats = Counter(stats)
        stats_dict.setdefault(accent, []).append(stats)

for k, _ in stats_dict.items():
    print(k)
    print(stats_dict[k][0])
    print(stats_dict[k][1])
    print(stats_dict[k][2])
    print()

with open('classifier_selection_stats.txt', 'w') as fp:
    for k, _ in stats_dict.items():
        fp.write(k)
        fp.write('\n')
        json.dump(stats_dict[k][0], fp)
        fp.write('\n')
        fp.write('\n')

f=open('/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/african/manifests/TSS_output/all/budget_100/classifier/run_1/train/train.json')
ff = [json.loads(line.strip()) for line in f]
type(ff[1])

print(ff[1])

d = {'a':[1], 'b':[2]}
print(d)
d.setdefault('a', []).append(3)
print(d)
d.setdefault('c', []).append(4)
print(d)

l = [1, 2, 1, 1, 3]
Counter(l)
