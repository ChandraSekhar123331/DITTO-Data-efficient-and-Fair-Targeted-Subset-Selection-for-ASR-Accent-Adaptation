declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 
# declare -a accents=('malayalam_male_english') 

target=20
feature='39'
b1=3500
b2=250

for accent in "${accents[@]}"
do
#     . scripts/tss.sh $accent $b1 $target $feature
	echo "______________________________________________________________________________"
	echo "$accent"
	. scripts/err_select.sh $accent $target $b1 $b2 $feature
	echo "____________________________________________________"
	echo
done

b1=2000
b2=250

for accent in "${accents[@]}"
do
#     . scripts/tss.sh $accent $b1 $target $feature
	echo "______________________________________________________________________________"
	echo "$accent"
	. scripts/err_select.sh $accent $target $b1 $b2 $feature
	echo "____________________________________________________"
	echo
done