declare -a accents=('indian' 'african')  
# declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 

target=25
feature='39'
b1=2500
b2=250


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
	. scripts/ts_pretrain.sh $accent $target $b1 $b2 $feature
    # . scripts/tss.sh $accent $b1 $target $feature > tss-logs.txt
	# . scripts/ts_top.sh $accent $target $b1 $b2 $feature
	. scripts/ts_rand.sh $accent $target $b1 $b2 $feature
	# . scripts/ts_err.sh $accent $target $b1 $b2 $feature
    # . scripts/random.sh $accent $b2
	echo "____________________________________________________"
	echo
	echo
done

