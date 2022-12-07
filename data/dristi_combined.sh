#declare -a speakers=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 
declare -a speakers=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english') 
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
	. dristi_run_random.sh $speaker
	. dristi_run.sh $speaker
	. dristi_run_within_random.sh $speaker
	echo "__________________________"
	echo
	echo
done

