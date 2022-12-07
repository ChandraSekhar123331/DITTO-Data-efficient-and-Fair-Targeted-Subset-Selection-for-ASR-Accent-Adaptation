#%%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from collections import Counter
from pathlib import Path

#%%

#%%
def load_features(accent, feature_type, file_name):
    features = []
    with open(file_name.replace('.json', f'_{feature_type}.file'), 'rb') as f:
        while True:
            try:
                features.append(pickle.load(f))
            except EOFError:
                break
            if features[-1].shape[1]<39:
                print(features[-1].shape)
                print(file_dir, "bad feature file")
                return np.array([[]])
                sys.exit()
    features = np.concatenate(features, axis=0)
    return features
    
#%%

#%%
list_dfs = []
accents = ['assamese_female_english', 'gujarati_female_english', 'hindi_male_english', 'kannada_male_english', 'malayalam_male_english', 'manipuri_female_english', 'rajasthani_male_english', 'tamil_male_english']
for accent in accents:
    file_name = f'{accent}/manifests/selection.json'
    feature_type = '39'

    features = load_features(accent, feature_type, file_name)
    df1 = pd.DataFrame(features)
    item_list = [dict(json.loads(line.strip()), **{'dir':accent}) for line in open(file_name)]
    df2 = pd.DataFrame(item_list)

    df = pd.concat([df1, df2], axis=1)
    print(len(df))
    list_dfs.append(df.copy())

df = pd.concat(list_dfs)
print(df.head())
print(len(df))

#%%

#%%
X = df.iloc[:, :39]
print(X.shape)

K = 8
kmeans = KMeans(K, random_state=0)
kmeans.fit(X)

#%%

#%%
cluster_ids = kmeans.fit_predict(X)
print(len(cluster_ids))
df['cluster_label'] = cluster_ids

#%%

#%%
#wcss = []
#for i in range(1, 12):
#    kmeans = KMeans(i)
#    kmeans.fit(X)
#    wcss_iter = kmeans.inertia_
#    wcss.append(wcss_iter)
#
#no_clusters = range(1, 12)
#plt.plot(no_clusters, wcss)
#plt.title('The Elbow title')
#plt.xlabel('No of clusters')
#plt.ylabel('WCSS')
#
#plt.savefig('clusters_elbow.png')

#%%

#%%
target = 'malayalam_male_english'
file_name = f'{target}/manifests/seed.json'
feature_type = '39'

features = load_features(accent, feature_type, file_name)
df_target = pd.DataFrame(features)

print(len(df_target))

#%%


#%%
dist_dict = {}
for k in range(K):
    df_temp = df.loc[df['cluster_label'] == k]
    x = df_temp.iloc[:, :39].to_numpy()
    y = df_target.to_numpy()
    dist_dict[k] = np.average(pairwise_distances(x, y))

#%%

#%%
for i in range(K):
    print(i, dist_dict[i])
closest_cluster = min(dist_dict, key=dist_dict.get)
print(closest_cluster)

#%%

#%%
dir_dist = df.loc[df['cluster_label'] == closest_cluster]['dir'].tolist()
selections = random.sample(dir_dist, 200)

print(Counter(selections))

#%%

#%%
df_sub = df.loc[df['cluster_label'] == closest_cluster]
print(df_sub)
for i in df_sub.sample(frac=1).iterrows():
    print(i[1]['text'])
    break

#%%

#%%
budget = 200
budget_dur = int(4.92*budget)
feature_type = '39'
stats_dict = {}
targets = ['assamese_female_english', 'gujarati_female_english', 'hindi_male_english', 'kannada_male_english', 'malayalam_male_english', 'manipuri_female_english', 'rajasthani_male_english', 'tamil_male_english']
for target in targets:
    file_name = f'{target}/manifests/seed.json'
    features = load_features(target, feature_type, file_name)
    df_target = pd.DataFrame(features)

    dist_dict = {}
    for k in range(K):
        df_temp = df.loc[df['cluster_label'] == k]
        x = df_temp.iloc[:, :39].to_numpy()
        y = df_target.to_numpy()
        dist_dict[k] = np.average(pairwise_distances(x, y))
    closest_cluster = min(dist_dict, key=dist_dict.get)
    df_sub = df.loc[df['cluster_label'] == closest_cluster]

    for i in range(1, 4):
        path = f'/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/data/{target}/all/budget_{budget}/cluster/run_{i}/train/'
        Path(path).mkdir(parents=True, exist_ok=True)
        stats = []
        with open(path+'train.json', 'w') as fp:
            total_dur = 0
            for row in df_sub.sample(frac=1).iterrows():
                if (total_dur + row[1]['duration']) <= budget_dur:
                    total_dur += row[1]['duration']
                    row_dict = {
                            'text': row[1]['text'],
                            'audio_filepath': row[1]['audio_filepath'],
                            'duration': row[1]['duration'],
                            'dir': row[1]['dir']
                            }
                    stats.append(row[1]['dir'])
                    json.dump(row_dict, fp)
                    fp.write('\n')
            stats = Counter(stats)
        stats_dict.setdefault(target, []).append(stats)

with open('cluster_selection_stats.txt', 'w') as fp:
    for k, _ in stats_dict.items():
        fp.write(k)
        fp.write('\n')
        json.dump(stats_dict[k][0], fp)
        fp.write('\n')
        json.dump(stats_dict[k][1], fp)
        fp.write('\n')
        json.dump(stats_dict[k][2], fp)
        fp.write('\n\n\n\n\n')

#%%
