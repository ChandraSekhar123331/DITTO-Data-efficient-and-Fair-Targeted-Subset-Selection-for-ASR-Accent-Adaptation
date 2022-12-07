# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english')
# declare -a accents=('rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
# declare -a accents=('hindi_male_english' 'assamese_female_english' 'gujarati_female_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english' 'manipuri_female_english')
declare -a accents=('hindi_male_english')
# declare -a functions=('GCMI' 'FL2MI' 'FL1MI' 'LogDMI')
declare -a functions=('FL2MI')
# declare -a functions=('FL1MI' 'LogDMI')
curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name


finetunepath=$(cd ../models/quartznet_asr; pwd)
homepath=$(pwd)


decayfactor=$1
eta_scale=1.0
ngram=2

echo $base_eta $eta_scale $ngram


target=20
budget=150
accent_similarity="cosine"
content_similarity="cosine"
# fxn="FL1MI"
accent_feature_type="39"
content_feature_type=tf_idf_"$ngram"gram



for accent in "${accents[@]}"
do
    for fxn in "${functions[@]}"
    do


        echo
        echo 
        echo 
        echo $accent $budget $target $accent_similarity $content_similarity $accent_feature_type $content_feature_type $fxn
        echo
        echo
        echo ngram $ngram eta_scale $eta_scale decayfactor $decayfactor


        # for run in 1 2 3
        for run in 1 2 3
        do 
            echo "---------------beginning finetuning----------------------------" 
            echo 
            echo 

            file_dir=$accent/all/dim_phoneme_gains/tau_$decayfactor/$accent=1/budget_$budget/target_$target/"$fxn"_etaScale_"$eta_scale"/accent_"$accent_feature_type"/content_"$content_feature_type"/phoneme_pseudo-phfreq-1gram/kernel_g=accent_gq=accent_qq=accent/accent_"$accent_similarity"/content_"$content_similarity"/run_"$run"/
            mkdir -p $file_dir
            cp $accent/all/dim_phoneme_gains/tau_$decayfactor/$accent=1/budget_$budget/target_$target/"$fxn"_etaScale_"$eta_scale"/accent_"$accent_feature_type"/content_"$content_feature_type"/phoneme_pseudo-phfreq-1gram/kernel_g=accent_gq=accent_qq=accent/accent_"$accent_similarity"/content_"$content_similarity"/train.json $file_dir
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










# echo 
# echo 

# for fxn in "${functions[@]}"
# do
#     for accent in "${accents[@]}"
#     do

#         echo
#         echo 
#         echo  fxn = $fxn and accent = $accent started
#         echo 
#         echo 
        

#         python TSS_generic.py \
#         --budget 150 --target 20 \
#         --eta_scale 1.0 \
#         --accent_similarity cosine \
#         --content_similarity cosine \
#         --fxn $fxn --accent $accent \
#         --accent_feature_type 39_3rep \
#         --content_feature_type tf_idf_2gram \
#         --data_kernel_type accent \
#         --query_data_kernel_type accent \
#         --query_query_kernel_type accent


#         echo 
#         echo 
#         echo  fxn = $fxn and accent = $accent end
#     done

# done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"