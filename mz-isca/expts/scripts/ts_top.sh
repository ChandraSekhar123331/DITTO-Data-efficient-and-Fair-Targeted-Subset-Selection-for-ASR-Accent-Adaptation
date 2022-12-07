curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 







homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
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
   echo "--------------------top-smi based-----------------------"
   echo "$fxn"
   echo "$accent" "$b1" "$tar" "$b2" "$ft"
   file_dir=$accent/all/budget_$b1/target_$tar/$fxn/eta_$eta/$sim/$ft
   echo $file_dir
   . scripts/top.sh $b2 $file_dir
   for run in 1 2 3
   do
       file_dir=$accent/all/budget_$b1/target_$tar/$fxn/eta_$eta/$sim/$ft/budget_$b2/top/run_$run
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       . mz-isca-scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . mz-isca-scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"









