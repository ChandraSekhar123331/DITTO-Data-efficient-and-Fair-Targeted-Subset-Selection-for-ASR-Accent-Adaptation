
curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 



eta_scale=$1
ngram=2

budget=150
target=20
accent_similarity="cosine"
content_similarity="cosine"
# fxn="LogDMI"
accent_feature_type="39_norm"
content_feature_type=tf_idf_"$ngram"gram
data_kernel_type="content"
query_data_kernel_type="accent"
query_query_kernel_type="content"






declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')
# declare -a functions=('FL1MI')
declare -a functions=('LogDMI')
# declare -a accents=('gujarati_female_english' 'assamese_female_english')
# declare -a accents=('kannada_male_english')
for accent in "${accents[@]}"
do
    for fxn in "${functions[@]}"
    do 
        echo
        echo 
        echo 
        # echo $accent $budget $target $accent_similarity $content_similarity $accent_feature_type $content_feature_type $fxn
        echo
        echo
        echo ngram $ngram eta_scale $eta_scale

        python TSS_generic.py \
        --budget $budget \
        --target $target \
        --eta_scale $eta_scale \
        --accent_similarity $accent_similarity \
        --content_similarity $content_similarity \
        --fxn $fxn \
        --accent $accent \
        --accent_feature_type $accent_feature_type \
        --content_feature_type $content_feature_type \
        --data_kernel_type $data_kernel_type \
        --query_data_kernel_type $query_data_kernel_type \
        --query_query_kernel_type $query_query_kernel_type 
    done
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
