# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 
declare -a accents=('assamese_female_english' 'gujarati_female_english') 
for accent in "${accents[@]}"
do
	echo "__________________________"
	echo "$accent"
	. run_random.sh $accent
	. run.sh $accent
	. run_within_random.sh $accent
	echo "__________________________"
	echo
	echo
done

