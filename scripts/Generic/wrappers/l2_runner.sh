curr_file_name=$(basename "$0")
echo $curr_file_name
LOG_DIR=../../logs
mkdir -p $LOG_DIR/$curr_file_name/
echo "logging", $curr_file_name

CUDA=3
DATASET=L2

declare -a accents=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese')

# for accent in "${accents[@]}"; do
#     echo $accent
#     echo "Merging seed and dev"
#     python -m preproc.seed_plus_dev --cuda $CUDA --dataset $DATASET --accent $accent
# done &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# for accent in "${accents[@]}"; do
#     echo "$accent started"
#     echo "Creating all.json started"
#     python -m preproc.create_all_json --dataset $DATASET --cuda $CUDA --accent $accent
#     echo "$accent end"
# done &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# for accent in ${accents[@]}; do
#     declare -a filenames=('selection.json' 'seed.json' 'test.json' 'dev.json' 'seed_plus_dev.json')
#     for filename in ${filenames[@]}; do
#         echo "Finding pretrain error rates"
#         echo $accent $filename
#         python -m utils.pretrain_out --cuda $CUDA --dataset $DATASET --accent $accent --json_name $filename
#         echo
#         echo "Adding json attributes WER, CER, pseudo_text"
#         echo $accent $filename
#         python -m preproc.add_json_attributes --cuda $CUDA --dataset $DATASET --accent $accent --json_name $filename --output_json_name $filename
#         echo
#     done
#     echo
# done &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# for accent in "${accents[@]}"; do
#     echo "$accent started"
#     CUDA_VISIBLE_DEVICES=$CUDA python -m features.MFCC-features --dataset $DATASET --cuda $CUDA --accent $accent --json_name all.json
#     echo "$accent done"
# done &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
