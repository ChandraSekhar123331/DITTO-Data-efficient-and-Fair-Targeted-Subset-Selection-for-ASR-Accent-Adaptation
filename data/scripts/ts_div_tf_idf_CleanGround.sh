homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
# acc/all/b_100/t_10/GCMI/39/b_50/top/run_1/train.json

stage_fxn=$8
stage2_sim=$7
ngram=$6
eta=1.0
sim="euclidean"
ft=$5
b2=$4
b1=$3 #b1 is going to be 'full_budget' only
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
   echo "--------------------TF-IDF based diversity clean-----------------------"
   echo "$fxn"
   echo "$accent" "$b1" "$tar" "$b2" "$ft"
   file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft
#    mkdir -p $file_dir
#    cp $accent/selection.json $file_dir/train.json
   . scripts/div_tf_idf.sh $accent $b2 $file_dir $file_dir/train.json $ngram $stage2_sim $stage2_fxn
   for run in 1 2 3
   do
       file_dir=$accent/all/budget_$b1/target_$tar/$fxn/$ft/budget_$b2/div_tf_idf_"$ngram"gram_"$stage2_sim"/"$stage2_fxn"/run_$run
#        mkdir -p file_dir
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       . scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
done
