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

# . scripts/tss.sh $accent $b1 $tar $ft
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
   echo
   echo
   echo "--------------------top-smi based-----------------------"
   echo "$fxn"
   echo "$accent" "$b1" "$tar" "$b2" "$ft"
   file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft
#    mkdir -p $file_dir
#    cp $accent/selection.json $file_dir/train.json
   . scripts/rand.sh $b2 $file_dir
   for run in 1 2 3
   do
       file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft/budget_$b2/random/run_$run
       mkdir -p file_dir
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       . scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
done
