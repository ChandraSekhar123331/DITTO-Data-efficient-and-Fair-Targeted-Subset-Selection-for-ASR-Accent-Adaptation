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

# {
#     python -m methods.global_random --dataset $DATASET --cuda $CUDA --budget 500 --sample
#     python -m methods.global_random --dataset $DATASET --cuda $CUDA --budget 500 --finetune --finetune_accent assamese
#     python -m methods.global_random --dataset $DATASET --cuda $CUDA --budget 500 --finetune --finetune_accent tamil
#     python -m methods.global_random --dataset $DATASET --cuda $CUDA --budget 500 --finetune --finetune_accent hindi
#     python -m methods.global_random --dataset $DATASET --cuda $CUDA --budget 500 --finetune --finetune_accent rajasthani

# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
# {
#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FL2MI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 20 --target_accent assamese-hindi::1-1 --sample --target_directory_path mixed

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn GCMI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 20 --target_accent assamese-hindi::1-1 --sample --target_directory_path mixed

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FL2MI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 20 --target_accent rajasthani-tamil::1-1 --sample --target_directory_path mixed

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn GCMI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 20 --target_accent rajasthani-tamil::1-1 --sample --target_directory_path mixed

# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# {
#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 750 --fxn FL2MI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 50 --target_accent assamese-hindi::1-1 --sample --target_directory_path mixed

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 750 --fxn GCMI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 50 --target_accent assamese-hindi::1-1 --sample --target_directory_path mixed

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 750 --fxn FL2MI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 50 --target_accent rajasthani-tamil::1-1 --sample --target_directory_path mixed

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 750 --fxn GCMI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 50 --target_accent rajasthani-tamil::1-1 --sample --target_directory_path mixed

# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# {
#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FL2MI --eta 1.0 --sim cosine --feature MFCC \
#         --target 20 --target_accent assamese-hindi::1-1 --target_directory_path mixed --sample

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn GCMI --eta 1.0 --sim cosine --feature MFCC \
#         --target 20 --target_accent assamese-hindi::1-1 --target_directory_path mixed --sample

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FL2MI --eta 1.0 --sim cosine --feature MFCC \
#         --target 20 --target_accent rajasthani-tamil::1-1 --target_directory_path mixed --sample

#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn GCMI --eta 1.0 --sim cosine --feature MFCC \
#         --target 20 --target_accent rajasthani-tamil::1-1 --target_directory_path mixed --sample

# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

{

    budget=4000
    eta=1.0
    fxns=('GCMI' 'FL2MI')
    feature=MFCC
    sims=('euclidean' 'cosine')
    target=20

    for sim in "${sims[@]}"; do
        for fxn in "${fxns[@]}"; do
            python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
                --budget $budget --fxn $fxn --eta $eta --sim $sim --feature $feature \
                --target $target --target_accent assamese-hindi::1-1 \
                --target_directory_path mixed --sample

            python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
                --budget $budget --fxn $fxn --eta $eta --sim $sim --feature $feature \
                --target $target --target_accent rajasthani-tamil::1-1 \
                --target_directory_path mixed --sample
        done
    done
} &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# {
#     python -m methods.global_TSS --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FL2MI --eta 1.0 --sim euclidean --feature MFCC \
#         --target 20 --target_accent assamese-hindi::1-1 --target_directory_path mixed --finetune --finetune_accent assamese

# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

# {
#     python -m methods.global_SM --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FacLoc --sim euclidean --lambdaVal 1 \
#         --feature MFCC --sample

#     python -m methods.global_SM --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn LogDet --sim euclidean --lambdaVal 1 \
#         --feature MFCC --sample

#     python -m methods.global_SM --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn FacLoc --sim cosine --lambdaVal 1 \
#         --feature MFCC --sample

#     python -m methods.global_SM --dataset $DATASET --cuda $CUDA \
#         --budget 500 --fxn LogDet --sim cosine --lambdaVal 1 \
#         --feature MFCC --sample
# } &>$LOG_DIR/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
