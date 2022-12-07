declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 

# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english') 


b=200

for accent in "${accents[@]}"
do
	echo "______________________________________________________________________________"
	echo "$accent"
	. scripts/random.sh $accent $b
	echo
	echo
done