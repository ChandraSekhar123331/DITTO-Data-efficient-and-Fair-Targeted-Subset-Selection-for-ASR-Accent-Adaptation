#declare -a other_accents=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese') 
declare -a other_accents=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese') 
main_accent=$1
for other_accent in "${other_accents[@]}"
do
	if [[ "$main_accent" != "$other_accent" ]];
	then
		echo "__________________________"
		echo "$main_accent" "$other_accent"
		. run_conditional.sh $main_accent $other_accent
		echo "__________________________"
		echo
		echo
	fi
done

