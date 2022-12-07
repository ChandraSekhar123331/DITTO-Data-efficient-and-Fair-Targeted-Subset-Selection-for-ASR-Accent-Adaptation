declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english')
# declare -a accents=('rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
# declare -a accents=('rajasthani_male_english')
# other_accent='tamil_male_english'
declare -a files=('seed.json' 'selection.json' 'test.json')
curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name


PYTH_script="../entropy-testing/pseudo-transcript-entropy/models/error_model/generate_phone_freq_featuers.py"


for accent in "${accents[@]}"
do
    for file_name in "${files[@]}"
    do


        echo
        echo 
        echo 
        echo "Start of $accent-$file_name experiment"
        echo
        echo

        python $PYTH_script \
        --selection_json_dir ./"$accent"/ \
        --selection_json_name "$file_name" 

        echo "End of $accent-$file_name experiment"
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