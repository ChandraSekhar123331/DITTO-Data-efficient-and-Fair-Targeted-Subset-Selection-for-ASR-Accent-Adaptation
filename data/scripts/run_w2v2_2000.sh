# declare -a accents=('tamil_male_english')  
declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 

stage2_sim='cosine'
ngram=3
target=20
feature='39'
b1=2000
b2=250


for accent in "${accents[@]}"
do
	echo "______________________________________________________________________________"
	echo "$accent"
#   . scripts/ts_w2v2_avg.sh $accent $target $b1 $b2 $feature
#   . scripts/ts_true_wer.sh $accent $target $b1 $b2 $feature
  . scripts/ts_div_tf_idf.sh $accent $target $b1 $b2 $feature $ngram $stage2_sim
	echo "____________________________________________________"
	echo
	echo
done