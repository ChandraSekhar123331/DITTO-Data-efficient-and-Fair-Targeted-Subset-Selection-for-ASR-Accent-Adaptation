import csv
import matplotlib.pyplot as plt

budget, target, features = 100, 10, '39'
# budget, target, features = 100, 50, 'TRILL'
csv_name = "report_{}_{}_{}.csv".format(budget, target, features)
# sheet_csv="sheet_{}_{}_{}.csv".format(budget, target, features)
WER_random={}
WER_function={}

infile = open(csv_name, 'r')
reader = csv.reader(infile)

# cols = ['speaker', 'ground', 'function', 'similarity', 'duration', 'samples', 
#         'WER-r1', 'WER-r2', 'WER-r3', 'WER-mean', 'WER-var', 'accents_run1', 'accents_run2', 'accents_run3', 'speakers_run1', 'speakers_run2', 'speakers_run3']

rows = [{k: v for k, v in row.items()} for row in csv.DictReader(infile)]
function_set = set()
for row in rows:
    speaker, ground, function, WER = row['speaker'], row['ground'], row['function'], float(row['WER-mean'])
    if ground=='within':
        continue
    if row['similarity']!='euclidean' and function!='random':
        continue
    function_set.add(function)
    if function=='random' and WER>0.01:
        WER_random[speaker]=WER
    else:
        if speaker not in WER_function:
            WER_function[speaker]={}
        WER_function[speaker][function]=WER

function_set.remove('random')
functions=list(function_set)
function_dict={}
for function in functions:
    function_dict[function]=[]
x=[]
y=[]
for speaker in WER_random:
    if abs(WER_random[speaker])<1e-1:
        print(speaker, " has no value for random")
        continue
    best=-20
    positives=0
    negatives=0
    for function in WER_function[speaker]:
        temp = -(WER_function[speaker][function]-WER_random[speaker])/WER_random[speaker]*100
        if temp<-15:
            print(speaker, function, WER_function[speaker][function], "random baseline: ",WER_random[speaker])
#             continue
        if temp<-5:
            print(speaker, function, " has loss {:.2f}% worse than random".format(-temp))
            pass
        if temp>0:
            positives+=1
        else:
            negatives+=1
        best=max(best,temp)
        function_dict[function].append(temp)
        y.append(temp)
    print("for speaker {}, wrt random: positives: {}, negatives: {}".format(speaker, positives, negatives))
    x.append(best)

p=sum(imp > 0 for imp in x)
n=sum(imp <= 0 for imp in x)

pos_all=sum(imp > 0 for imp in y)
neg_all=sum(imp <= 0 for imp in y)

print("improvements over random {} out of {} times".format(pos_all, pos_all+neg_all))

for function in functions:
    print("for {}: positives: {}, negatives: {}".format(function, 
                                                        sum(imp > 0 for imp in function_dict[function]),
                                                        sum(imp <= 0 for imp in function_dict[function])))


for func in function_dict:
    print(func, " {:.2f}%".format(sum(function_dict[func])/len(function_dict[func])))

function_set.add('random')
function_time=dict(zip(list(function_set), [0]*len(list(function_set))))
for row in rows:
    speaker, ground, function, WER = row['speaker'], row['ground'], row['function'], float(row['WER-mean'])
    if ground=='within':
        continue
    if row['similarity']!='euclidean' and function!='random':
        continue
    function_time[function]+=WER
    
print('random', " {:.2f}".format(function_time['random']/24))
for func in function_dict:
    print(func, " {:.2f}".format(function_time[func]/24))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.title.set_text('taking best function for each speaker')
ax2.title.set_text('taking all functions')
ax1.hist(x, bins=10)
ax2.hist(y, bins=25)
print("saving best MI histograms in "+'hist_best_{}_{}_{}.png'.format(budget, target, features))
fig.savefig('hist_best_{}_{}_{}.png'.format(budget, target, features), bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
plt.tight_layout()
ax1.title.set_text(functions[0])
ax2.title.set_text(functions[1])
ax3.title.set_text(functions[2])
ax4.title.set_text(functions[3])
ax1.hist(function_dict[functions[0]], bins=10)
ax2.hist(function_dict[functions[1]], bins=10)
ax3.hist(function_dict[functions[2]], bins=10)
ax4.hist(function_dict[functions[3]], bins=10)
print("saving individual functions' histograms in "+'hist_indiv_{}_{}_{}.png'.format(budget, target, features))
fig.savefig('hist_indiv_{}_{}_{}.png'.format(budget, target, features), bbox_inches='tight')


infile.close()
