curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 

finetunepath=$(cd ../models/quartznet_asr; pwd)
homepath=$(pwd)


base_eta=1.0
accent=$3
eta_scale=$2
ngram=$1

echo $base_eta $eta_scale $ngram $accent


target=20
budget=150
accent_similarity="cosine"
content_similarity="cosine"
fxn="FL1MI"
accent_feature_type="39_norm"
content_feature_type=tf_idf_"$ngram"gram




    echo
    echo 
    echo 
    echo $accent $budget $target $accent_similarity $content_similarity $accent_feature_type $content_feature_type $fxn
    echo
    echo
    echo ngram $ngram eta_scale $eta_scale


    # for run in 1 2 3
    for run in 1 2 3
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
    done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
