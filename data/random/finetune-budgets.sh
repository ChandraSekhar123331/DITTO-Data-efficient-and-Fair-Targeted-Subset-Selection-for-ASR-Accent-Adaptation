# for budget in 100 200
for budget in 200
do
   echo
   echo "-----------------finetuning random selections--------------------"
   echo "$budget"
   for run in 1 2 3
   do
       file_dir=random/$budget/run_$run
       mkdir -p file_dir
       echo "---------------beginning finetuning----------------------------"
       echo $file_dir
       cd "$finetunepath"
       . scripts/finetune.sh $accent $file_dir
       cd "$homepath"
   done
done