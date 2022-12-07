budget, target, features = 100, 5, '39'
# budget, target, features = 100, 50, 'TRILL'
csv_name = "report_{}_{}_{}.csv".format(budget, target, features)
sheet_csv="sheet_{}_{}_{}-with.csv".format(budget, target, features)

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
    elif line.split(',')[1]!=ground:
        ground=line.split(',')[1]
        sheet.write('\n')
    sheet.write(line)

sheet.close()
original.close()
