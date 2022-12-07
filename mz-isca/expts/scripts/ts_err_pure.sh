curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 







homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
# acc/all/b_100/t_10/GCMI/39/b_50/top/run_1/train.json
b2=$2
accent=$1

{
   echo
   echo
   echo "--------------------error based fully within-----------------------"
   echo "$accent" "$b2"

   mkdir -pv $accent/all/within/
   cp -v $accent/selection.json $accent/all/within/train.json

   file_dir=$accent/all/within/
   . scripts/error.sh $accent $b2 $file_dir $file_dir/train.json
   for run in 1 2 3
   do
         file_dir=$accent/all/within/budget_$b2/error_model/run_$run
         echo "---------------beginning finetuning----------------------------"
         cd "$finetunepath"
         mkdir -pv $file_dir
         . mz-isca-scripts/finetune.sh $accent $file_dir
         echo "---------------beginning testing----------------------------"
         . mz-isca-scripts/test.sh $accent $file_dir
         cd "$homepath"
   done
} &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
