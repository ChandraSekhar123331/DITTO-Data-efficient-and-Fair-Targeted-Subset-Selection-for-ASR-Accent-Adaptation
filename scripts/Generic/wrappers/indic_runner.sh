curr_file_name=$(basename "$0")
echo $curr_file_name
LOG_DIR=../../logs
mkdir -p $LOG_DIR/$curr_file_name/
echo "logging", $curr_file_name

CUDA=0
DATASET=INDIC

declare -a accents=('kannada' 'rajasthani' 'gujarati' 'hindi' 'malayalam' 'assamese' 'manipuri' 'tamil')

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

# {
#     python -m preproc.prep_mixed_set \
#         --dataset $DATASET --cuda $CUDA \
#         --query_set assamese hindi \
#         --query_set_composn 1 1

#     python -m preproc.prep_mixed_set \
#         --dataset $DATASET --cuda $CUDA \
#         --query_set assamese hindi \
#         --query_set_composn 3 5

#     python -m preproc.prep_mixed_set \
#         --dataset $DATASET --cuda $CUDA \
#         --query_set rajasthani tamil \
#         --query_set_composn 1 1

# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
