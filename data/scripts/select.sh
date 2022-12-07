homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
# acc/all/b_100/t_10/GCMI/39/b_50/top/run_1/train.json
eta=1.0
sim="euclidean"
ft=$5
b2=$4
b1=$3
tar=$2
accent=$1

. scripts/tss.sh $accent $b1 $tar $ft
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
   echo "--------------------error based-----------------------"
   echo "$fxn"
   echo "$accent" "$b1" "$tar" "$b2" "$ft"
   file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft
   . scripts/rand.sh $b2 $file_dir
   . scripts/top.sh $b2 $file_dir
   . scripts/true_wer.sh $b2 $file_dir $file_dir/train_wers_appended.json
#    . scripts/error.sh $accent $b2 $file_dir $file_dir/train.json
done


echo GLOBAL RANDOM
if [ ! -d random/$budget/run_1/ ]; then
  echo reached
  python select-random.py  --budget $budget --file_dir /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/data/random/all_selection.json
fi

for run in 1 2 3
do
   echo "--------------------full random-----------------------"
   echo "$accent" "$budget"
   data_dir=$accent/all/budget_$budget/random/run_$run
   mkdir -p $data_dir
   cp random/$budget/run_$run/train.json $data_dir/train.json
done
