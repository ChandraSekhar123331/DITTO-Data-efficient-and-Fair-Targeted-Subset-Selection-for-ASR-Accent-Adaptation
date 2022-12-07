#%%
import random
import json

def create(json_path):
    json_file = open(json_path)
    json_item_list = [line for line in json_file]
    random.shuffle(json_item_list)
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    return json_item_list

#%%

#%%
s1 = create('../hindi_male_english/dev.json')
s2 = create('../rajasthani_male_english/dev.json')

s1 = s1[:len(s1)//2]
s2 = s2[:len(s2)//2]
s = s1 + s2

with open('hindi_rajasthani/dev_mix.json', 'w') as f:
    for line in s:
        f.write('{}\n'.format(json.dumps(line)))
#%%
