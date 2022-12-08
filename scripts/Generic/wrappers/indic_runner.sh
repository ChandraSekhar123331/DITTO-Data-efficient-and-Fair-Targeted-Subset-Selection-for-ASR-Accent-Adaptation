curr_file_name=$(basename "$0")
echo $curr_file_name
mkdir -p logs/$curr_file_name/
echo "logging",  $curr_file_name
echo 
echo 

CUDA=0
DATASET=INDIC

declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'malayalam_male_english' 'assamese_female_english' 'manipuri_female_english' 'tamil_male_english')

for accent in "${accents[@]}"
do
echo $accent
python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name selection.json
python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed.json
python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name test.json
python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name dev.json
python -u pretrain_out.py --cuda $CUDA --dataset $DATASET --accent $accent --json_name seed_plus_dev.json
echo
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"



