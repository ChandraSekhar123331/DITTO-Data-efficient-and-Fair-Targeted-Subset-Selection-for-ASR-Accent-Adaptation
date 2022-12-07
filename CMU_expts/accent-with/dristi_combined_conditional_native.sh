declare -a main_accents=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese') 
for main_accent in "${main_accents[@]}"
do
	echo "__________________________"
	echo "$main_accent" 
	. dristi_run_conditional_native.sh $main_accent 
	echo "__________________________"
	echo
	echo
done

