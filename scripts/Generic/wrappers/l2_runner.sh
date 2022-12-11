curr_file_name=$(basename "$0")
echo $curr_file_name
mkdir -p logs/$curr_file_name/
echo "logging",  $curr_file_name
echo 
echo 

CUDA=2
DATASET=L2

declare -a accents=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese') 

# for accent in "${accents[@]}"
# do
# echo $accent
# echo "Merging seed and dev"
# python -u seed_plus_dev.py --cuda $CUDA --dataset $DATASET --accent $accent
# done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"



# for accent in "${accents[@]}"
# do
# echo $accent
# # python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name selection.json
# # python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed.json
# # python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name test.json
# # python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name dev.json
# # python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed_plus_dev.json
# echo
# done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"


# for accent in "${accents[@]}"
# do
#     echo accent $accent dataset $DATASET
#     echo "Adding attributes to jsons"
#     python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name selection.json --output_json_name selection.json
#     python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed.json --output_json_name seed.json
#     python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name test.json --output_json_name test.json
#     python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name dev.json --output_json_name dev.json
#     python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed_plus_dev.json --output_json_name seed_plus_dev.json
#     echo "done"
# done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"


for accent in "${accents[@]}"
do
    echo accent $accent dataset $DATASET
    echo "Training error model"
    python -u error.py --cuda $CUDA --dataset $DATASET --accent $accent 
    python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed.json --output_json_name seed.json
    python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name test.json --output_json_name test.json
    python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name dev.json --output_json_name dev.json
    python -u add_json_attributes.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed_plus_dev.json --output_json_name seed_plus_dev.json
    echo "done"
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"


