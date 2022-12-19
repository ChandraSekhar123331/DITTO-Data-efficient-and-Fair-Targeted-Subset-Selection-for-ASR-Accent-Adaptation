curr_file_name=$(basename "$0")
echo $curr_file_name
mkdir -p logs/$curr_file_name/
echo "logging",  $curr_file_name
echo 
echo 

CUDA=3
DATASET=MCV


# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
# Issues are there with bermuda, wales, and malaysia.. Ignore those
declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'southatlandtic')
# for accent in "${accents[@]}"
# do
# echo $accent
# python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name selection.json
# python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed.json
# python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name test.json
# python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name dev.json
# python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed_plus_dev.json
# echo
# done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"


for accent in "${accents[@]}"
do
echo "$accent started"
CUDA_VISIBLE_DEVICES=$CUDA python -m features.MFCC-features --dataset $DATASET --cuda $CUDA --accent $accent --json_name all.json
echo "$accent done"
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"



