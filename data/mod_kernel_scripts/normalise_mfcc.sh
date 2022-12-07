declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english')

declare -a filenames=('seed' 'selection' 'test')

curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 

for accent in "${accents[@]}"
do
    for filename in "${filenames[@]}"
    do

                
        echo 
        echo 
        inp_feature_type="39"
        outp_feature_type="39_norm"

        inp_file_name=$filename
        inp_dir="$accent"/"$inp_feature_type"
        outp_dir="$accent"/"$outp_feature_type"

        echo "accent: ", $accent
        echo "filename: ", $filename
        echo "inp_feature_type: ", $inp_feature_type
        echo "outp_feature_type: ", $outp_feature_type
        echo "input_file_name: ", $inp_file_name
        echo "input_dir: ", $inp_dir
        echo "output_dir: ", $outp_dir

        python normalise_mfcc.py \
        --outp_feature_type $outp_feature_type \
        --inp_feature_type $inp_feature_type \
        --input_file_name $inp_file_name \
        --input_dir $inp_dir \
        --output_dir $outp_dir 


        echo 
        echo 
    done

done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"