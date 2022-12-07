#declare -a speakers=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese') 
declare -a speakers=('korean' 'spanish' 'vietnamese') 
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
#	. run_random.sh $speaker
#	. run.sh $speaker
	. run_within_equal_random.sh $speaker
	echo "__________________________"
	echo
	echo
done

