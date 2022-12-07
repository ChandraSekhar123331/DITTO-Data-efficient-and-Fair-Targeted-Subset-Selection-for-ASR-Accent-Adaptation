finetunepath=$(cd ../models/quartznet_asr; pwd)
homepath=$(pwd)


base_eta=$3
eta_scale=$2
ngram=$1

echo $base_eta $eta_scale $ngram


target=20
budget=150
accent_similarity="euclidean"
content_similarity="euclidean"
fxn="FL1MI"
accent_feature_type="39"
content_feature_type=tf_idf_"$ngram"gram




# # declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
declare -a accents=('assamese_female_english')

for accent in "${accents[@]}"
do
    echo
    echo 
    echo 
    echo $accent $budget $target $accent_similarity $content_similarity $accent_feature_type $content_feature_type $fxn
    echo
    echo
    echo ngram $ngram eta_scale $eta_scale


    # for run in 1 2 3
    for run in 1
    do 
        echo "---------------beginning finetuning----------------------------" 
        echo 
        echo 

        file_dir=$accent/all/budget_$budget/target_$target/"$fxn"_etabase_"$base_eta"_etaScale_"$eta_scale"/accent_"$accent_feature_type"/content_"$content_feature_type"/accent_"$accent_similarity"/content_"$content_similarity"/run_"$run"/
        mkdir -p $file_dir
        cp $accent/all/budget_$budget/target_$target/"$fxn"_etabase_"$base_eta"_etaScale_"$eta_scale"/accent_"$accent_feature_type"/content_"$content_feature_type"/accent_"$accent_similarity"/content_"$content_similarity"/train.json $file_dir
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
