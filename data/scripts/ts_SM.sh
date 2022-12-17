homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
# acc/all/b_100/t_10/GCMI/39/b_50/top/run_1/train.json
sim="euclidean"
fxn="FacLoc"
ft="39"
budget=250
accent=$1

# . scripts/tss.sh $accent $b1 $tar $ft
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
# for fxn in "${fxns[@]}"
# do
   echo
   echo
   echo "--------------------Submodular Selection(no-targeting)-----------------------"
   echo "$fxn"
#    echo "$accent" "$b1" "$tar" "$b2" "$ft"
#    file_dir=$accent/all/budget_$budget/$fxn/$sim
#    mkdir -p $file_dir
#    cp $accent/selection.json $file_dir/train.json
   python SM_select.py --fxn $fxn --accent $accent --features $ft --similarity $sim --budget $budget
   for run in 1 2 3
   do
       file_dir=$accent/all/budget_$budget/SM_select/$fxn/$ft/$sim/run_$run
       mkdir -pv $file_dir
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       . scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
# done
