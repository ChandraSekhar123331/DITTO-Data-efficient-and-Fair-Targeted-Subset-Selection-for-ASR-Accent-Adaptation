# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
declare -a accents=('hindi_male_english')
# declare -a accents=('rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
declare -a functions=('GCMI' 'FL2MI' 'FL1MI' 'LogDMI')
# declare -a functions=('FL1MI' 'LogDMI')
curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 

for fxn in "${functions[@]}"
do
    for accent in "${accents[@]}"
    do

        echo
        echo 
        echo  fxn = $fxn and accent = $accent started
        echo 
        echo 
        

        python TSS_generic.py \
        --budget 75 --target 20 \
        --eta_scale 1.0 \
        --accent_similarity cosine \
        --content_similarity cosine \
        --fxn $fxn --accent $accent \
        --accent_feature_type 39 \
        --content_feature_type tf_idf_2gram \
        --data_kernel_type accent \
        --query_data_kernel_type accent \
        --query_query_kernel_type accent


        echo 
        echo 
        echo  fxn = $fxn and accent = $accent end
    done

done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"