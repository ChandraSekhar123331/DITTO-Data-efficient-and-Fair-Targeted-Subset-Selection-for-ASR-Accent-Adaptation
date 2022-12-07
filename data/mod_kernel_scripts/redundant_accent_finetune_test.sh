curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 

finetunepath=$(cd ../models/quartznet_asr; pwd)
homepath=$(pwd)

# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english')

declare -a accents=()
declare -a accents=('kannada_male_english' 'assamese_female_english' 'manipuri_female_english' 'hindi_male_english')
# declare -a functions=('GCMI' 'FL2MI')
declare -a functions=('FL1MI' 'LogDMI')


# base_eta=1.0
# accent=$3
eta_scale=1.0
ngram=2

# echo $eta_scale $ngram


target=20
budget=150
accent_similarity="cosine"
content_similarity="cosine"
# fxn="FL1MI"
accent_feature_type="39_3rep"
content_feature_type=tf_idf_"$ngram"gram

data_kernel_type=accent
query_data_kernel_type=accent
query_query_kernel_type=accent

for fxn in "${functions[@]}"
do
    for accent in "${accents[@]}"
    do
        echo
        echo $accent $fxn
        echo 
        echo $accent $budget $target $accent_similarity $content_similarity $accent_feature_type $content_feature_type $fxn $data_kernel_type $query_data_kernel_type $query_query_kernel_type
        echo
        echo
        echo ngram $ngram eta_scale $eta_scale


        # for run in 1 2 3
        for run in 1 2 3
        do 
            echo "---------------beginning finetuning----------------------------" 
            echo $run start
            echo 
            echo 

            file_dir="$accent"/all/budget_"$budget"/target_"$target"/"$fxn"_etaScale_"$eta_scale"/accent_"$accent_feature_type"/content_"$content_feature_type"/kernel_g="$data_kernel_type"_gq="$query_data_kernel_type"_qq="$query_query_kernel_type"/accent_"$accent_similarity"/content_"$content_similarity"/run_"$run"
            echo file_dir is $file_dir
            echo MAking file_dir
            mkdir -pv $file_dir
            cp -v "$accent"/all/budget_"$budget"/target_"$target"/"$fxn"_etaScale_"$eta_scale"/accent_"$accent_feature_type"/content_"$content_feature_type"/kernel_g="$data_kernel_type"_gq="$query_data_kernel_type"_qq="$query_query_kernel_type"/accent_"$accent_similarity"/content_"$content_similarity"/train.json $file_dir
            cd "$finetunepath"
            . scripts/finetune.sh $accent $file_dir
            echo "---------------beginning testing----------------------------"
            echo 
            echo
            . scripts/test.sh $accent $file_dir
            cd "$homepath"

            echo done with run = $run file_dir = $file_dir
        done 
    done
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"

