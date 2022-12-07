import os, sys
budget, target, features = 200, 20, 'wv10_100'

setting = sys.argv[1]
csv_name = "{}_{}_{}_{}.csv".format(setting, budget, target, features)
sheet_csv="mz-isca-{}_{}_{}_{}.csv".format(setting, budget, target, features)


original = open(csv_name, 'r')
lines = original.readlines()
sheet = open(sheet_csv, 'w')
sheet.write(lines[0])
sheet.write(lines[1])
speaker = lines[1].split(',')[0]
ground = lines[1].split(',')[1]

for line in lines[2:]:
    if line.split(',')[0]!=speaker:
        speaker=line.split(',')[0]
        ground=line.split(',')[1]
        sheet.write('\n\n')
    sheet.write(line)

sheet.close()
original.close()


