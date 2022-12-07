DATA=$(cd /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/data; pwd)
file_dir=$2
budget=$1
# echo $accent $b2 $file_dir $file_dir/train.json


echo doing w2v2_avg features experiment.

IN=$file_dir
arrIN=(${IN//_english/ })
expt_details=${arrIN[1]}  
echo $expt_details
# for run in 1 2 3
# do

 
 
python3 -u stage2_TSS_w2v2.py \
      --file_dir=$file_dir \
      --budget=$budget \
# done
