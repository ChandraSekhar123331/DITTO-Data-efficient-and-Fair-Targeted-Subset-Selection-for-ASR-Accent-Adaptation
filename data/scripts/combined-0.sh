# declare -a accents=('assamese_female_english' 'gujarati_female_english')  
declare -a accents=('manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english' 'assamese_female_english' 'gujarati_female_english')


# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 
# decay_factor=$1
target=20
feature='39'
b1=3500
b2=$1
# stage2_fxn='LogDet'


# for accent in "${accents[@]}"
# do
# 	echo "______________________________________________________________________________"
# 	echo "$accent" selections
#   . scripts/select.sh $accent $target $b1 $b2 $feature
# 	echo "____________________________________________________"
# 	echo
# 	echo
# done


for accent in "${accents[@]}"
do
	echo "______________________________________________________________________________"
	echo "$accent"
#     . scripts/tss.sh $accent $b1 $target $feature > tss-logs.txt
# 	. scripts/ts_top.sh $accent $target $b1 $b2 $feature
# 	. scripts/ts_rand.sh $accent $target $b1 $b2 $feature
	. scripts/ts_err.sh $accent $target $b1 $b2 $feature
#     . scripts/ts_true_wer.sh $accent $target $b1 $b2 $feature
#     . scripts/random.sh $accent $b2
# 	. scripts/ts_div_tf_idf.sh $accent $target $b1 $b2 $feature 2 euclidean
#     . scripts/ts_div_tf_idf_CleanGround.sh $accent $target $b1 $b2 $feature 2 euclidean $stage2_fxn
    # . scripts/ts_uniform.sh $accent $target $b1 $b2 $feature $decay_factor
	echo "____________________________________________________"
	echo
	echo
done

