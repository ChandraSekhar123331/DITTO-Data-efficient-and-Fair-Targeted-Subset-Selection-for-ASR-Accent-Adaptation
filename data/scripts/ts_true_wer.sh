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
declare -a fxns=('FL2MI')
for fxn in "${fxns[@]}"
do
   echo
   echo
   echo "--------------------True WER based second level selections-----------------------"
   echo "$fxn"
   echo "$accent" "$b1" "$tar" "$b2" "$ft"
   
   cd "$finetunepath"
   file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft
   . scripts/chandra_pretrained_test.sh $accent $file_dir

    cd "$homepath"
    
    python3 -u append_wers_to_json.py --file_dir $file_dir

    . scripts/true_wer.sh $b2 $file_dir $file_dir/train_wers_appended.json
   for run in 1 2 3
   do
       file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft/budget_$b2/true_wer/run_$run
       mkdir -p file_dir
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       . scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
done
