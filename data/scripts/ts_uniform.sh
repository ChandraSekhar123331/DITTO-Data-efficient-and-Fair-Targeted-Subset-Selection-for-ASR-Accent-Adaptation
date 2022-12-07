homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
# acc/all/b_100/t_10/GCMI/39/b_50/top/run_1/train.json

eta=1.0
sim="euclidean"
decay_factor=$6
ft=$5
b2=$4
b1=$3
tar=$2
accent=$1

# . scripts/tss.sh $accent $b1 $tar $ft
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
declare -a fxns=('FL2MI')
# declare -a fxns=('GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
   echo
   echo
   echo "--------------------Uniform based diversity-----------------------"
   echo "$fxn"
   echo "$accent" "$b1" "$tar" "$b2" "$ft"
   file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft
   echo file_dir is $file_dir
   . scripts/uniform.sh $accent $b2 $file_dir $file_dir/train.json $decay_factor
   for run in 1 2 3
   do
       file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft/budget_$b2/uniform_$decay_factor/run_$run
    #    mkdir -p $file_dir
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       . scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
done
