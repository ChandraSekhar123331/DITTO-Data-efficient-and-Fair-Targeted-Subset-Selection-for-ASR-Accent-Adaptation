#%%
from collections import defaultdict

f = open('speaker-info.txt')

d = defaultdict(list)

for line in f.readlines():
    if 'AGE' in line:
        continue
    l = line.split()
    d['spkr'].append(l[0])
    d['age'].append(l[1])
    d['gender'].append(l[2])
    d['accent'].append(l[3])
    try: 
        temp = ' '.join(l[4:])
        if '(' in temp:
            a, b = temp.split('(')
            d['region'].append(a)
            d['comment'].append(b[:-1])
        else:
            d['region'].append(temp)
            d['comment'].append('')
    except:
        d['region'].append('')
        d['comment'].append('')

#%%

##%%
#import pandas as pd
#
#df = pd.read_csv('modified_speaker-info.txt')
#df.head(10)
#
##%%

#%%
from collections import Counter

print('Age distribution: ')
print(Counter(d['age']))
print('\nGender distribution: ')
print(Counter(d['gender']))
print('\nAccent distribution: ')
print(Counter(d['accent']))
#print(Counter(d['region']))

#%%

#%%
accent2region = defaultdict(list)

for a, r in zip(d['accent'], d['region']):
    accent2region[a].append(r)

print('\n\nRegion distribution')
for a, r in accent2region.items():
    print(a)
    print(r)
    print()
#%%
