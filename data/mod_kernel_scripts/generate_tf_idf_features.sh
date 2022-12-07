feature_generator=$(cd ../entropy-testing/pseudo-transcript-entropy/models/error_model/ ; pwd)
# base_dir=$(pwd)
ngram=$1
# echo  Hello

declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 
declare -a filetypes=('all' 'dev' 'seed_plus_dev' 'seed' 'selection' 'test') 

for accent in "${accents[@]}"
do
    for file_type in "${filetypes[@]}"
    do
        echo "$accent" "$file_type".json
        python "$feature_generator"/generate_tf_idf_features.py \
        --inp_json $accent/"$file_type".json \
        --ngram $ngram \
        --output_feature_file $accent/tf_idf_"$ngram"gram/"$file_type"_tf_idf_"$ngram"gram.file > $accent/tf_idf_"$ngram"gram/"$file_type"_log.txt
    done

done
